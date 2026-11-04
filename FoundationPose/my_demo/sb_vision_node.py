#!/usr/bin/env python3
"""Publish measured RGB optical-frame bottle poses to sb_communication/ObjectPose."""
from __future__ import annotations

import argparse
from collections import Counter, deque
import json
import os
import sys
from pathlib import Path
import threading
import time
import numpy as np

from async_pose_worker import LatestPoseRegistrationWorker
from sb_pose_contract import (RGBDSynchronizer, PublicationGate, load_object_profile, profile_target_class,
                              quaternion_xyzw)
from sb_vision_pipeline import FoundationPoseBackend, PosePipeline, PosePipelineError
from vision_metrics import VisionMetrics, ProcessGpuSampler, write_summary
from rgbd_transport import image_topics
from sequence_store import PersistentSequence, SequenceStateError
from clock_sync import (add_clock_arguments, settings_from_args, ClockSyncMonitor,
                        ClockEpochGuard, sample_clock_pair)

ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    # 默认目标瓶子在这里改：配置里的 class_id 决定 YOLO 类别，mesh 决定 FoundationPose 模型，两者总是一致。
    p.add_argument("--object-profile", type=Path, default=ROOT / "my_demo/config/dfsy.json",
                   help="物体配置；农夫山泉用 my_demo/config/nfsq.json，东方树叶用 my_demo/config/dfsy.json，"
                        "旧瓶子用 my_demo/config/bottle_current.json")
    p.add_argument("--yolo-model", type=Path, default=ROOT / "runs/yolov8/drink_seg_5cls.pt")
    p.add_argument("--target-class", type=int, default=None,
                   help="YOLO 类别：0=fenda, 1=cola, 2=qixi, 3=nfsq, 4=dfsy；默认取物体配置里的 class_id，没有则为 0")
    p.add_argument("--detector-confidence", type=float, default=.30)
    p.add_argument("--detector-image-size", type=int, default=960)
    p.add_argument("--detector-hz", type=float, default=10.0,
                   help="YOLO refresh rate while FoundationPose tracks every RGB-D frame")
    p.add_argument("--miss-grace-seconds", type=float, default=.30,
                   help="retain the last target association for this long after a detector miss")
    p.add_argument("--reselect-after-seconds", type=float, default=1.0,
                   help="wait for this long before selecting a different detected target")
    p.add_argument("--preload-backend", action=argparse.BooleanOptionalAction, default=True,
                   help="load YOLO/FoundationPose before processing the first camera frame")
    p.add_argument("--device", default="0")
    p.add_argument("--camera-prefix", default="/head_camera")
    p.add_argument("--image-transport", choices=("raw", "compressed"), default="compressed")
    p.add_argument("--frame-id", default="head_d435i_optical_frame")
    p.add_argument("--object-pose-topic", default="/object_pose")
    p.add_argument("--node-name", default="sb_foundationpose")
    p.add_argument("--status-topic", default="/vision/status")
    p.add_argument("--sequence-state-dir", type=Path,
                   default=Path(os.environ.get("SB_VISION_STATE_DIR") or ROOT / ".state/sb_vision"),
                   help="Persistent local sequence storage; keep this directory across restarts/deployments")
    p.add_argument("--sequence-start-after", type=int, default=None,
                   help="One-time initialization/migration floor: 0 for a new receiver, or its last accepted sequence")
    p.add_argument("--max-rgbd-skew-ms", type=float, default=20)
    p.add_argument("--max-observation-delay", type=float, default=.30)
    p.add_argument("--register-views", type=int, default=12)
    p.add_argument("--register-topk", type=int, default=12)
    p.add_argument("--register-iterations", type=int, default=3)
    p.add_argument("--track-iterations", type=int, default=2)
    p.add_argument("--pose-max-side", type=int, default=640, help="FP input longest side; K is rescaled with pixels")
    p.add_argument("--max-runtime-seconds", type=float, default=None)
    p.add_argument("--metrics-jsonl", type=Path, default=None)
    p.add_argument("--metrics-summary", type=Path, default=None)
    p.add_argument("--metrics-source", choices=("live_camera", "replay", "synthetic", "unspecified"), default="unspecified")
    p.add_argument("--gpu-memory-sample-ms", type=float, default=50.)
    add_clock_arguments(p)
    return p


def make_node(args, profile, backend_factory=None, clock_factory=None):
    args.target_class = profile_target_class(profile, args.target_class)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo, CompressedImage
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
    from sb_communication.msg import ObjectPose

    class VisionNode(Node):
        def __init__(self):
            super().__init__(args.node_name)
            self.sequence_store = None
            self.worker = None
            self.clock_monitor = None
            self.metrics = None
            self.performance = None
            self.gpu_sampler = None
            self.metrics_error = None
            self._closing = False
            try:
                self._initialize()
            except BaseException:
                try:
                    self.close()
                finally:
                    self.destroy_node()
                raise

        def _initialize(self):
            if self.get_parameter("use_sim_time").value:
                raise ValueError("this real-camera entry point requires system time")
            self.sync = RGBDSynchronizer(args.frame_id, args.max_rgbd_skew_ms)
            self.pub = self.create_publisher(ObjectPose, args.object_pose_topic, 10)
            self.sequence_store = PersistentSequence(args.sequence_state_dir,
                domain_id=self.context.get_domain_id(), topic=self.pub.topic_name,
                start_after=args.sequence_start_after)
            self.gate = PublicationGate(args.max_observation_delay, sequence_allocator=self.sequence_store)
            self.status_pub = self.create_publisher(String, args.status_topic, 10)
            self.pipeline = None
            self.reset_requested = threading.Event()
            self.reset_lock = threading.Lock()
            self.pending_reset = None
            self.performance = VisionMetrics({"input_source": args.metrics_source,
                "backend": "injected" if backend_factory else "FoundationPoseBackend",
                "domain_id": self.context.get_domain_id(), "object_pose_topic": self.pub.topic_name,
                "frame_id": args.frame_id, "image_transport": args.image_transport,
                "mesh_sha256": profile["mesh_sha256"], "target_class": args.target_class,
                "detector_confidence": args.detector_confidence, "detector_image_size": args.detector_image_size,
                "detector_hz": args.detector_hz, "miss_grace_seconds": args.miss_grace_seconds,
                "reselect_after_seconds": args.reselect_after_seconds,
                "preload_backend": args.preload_backend,
                "register_views": args.register_views, "register_topk": args.register_topk,
                "register_iterations": args.register_iterations, "track_iterations": args.track_iterations,
                "pose_max_side": args.pose_max_side, "max_observation_delay_s": args.max_observation_delay})
            self.gpu_sampler = ProcessGpuSampler(args.gpu_memory_sample_ms / 1000.)
            self.worker = LatestPoseRegistrationWorker(self.infer, name="foundationpose-inference")
            self.latest = None
            self.counts = Counter()
            self.latencies = deque(maxlen=512)
            self.last_result = {}
            self.fatal_error = None
            self.start_ns = self.get_clock().now().nanoseconds
            self.reset_cutoff_ns = 0
            self.clock_settings = settings_from_args(args)
            self.clock_monitor = (clock_factory or ClockSyncMonitor)(self.clock_settings)
            self.clock_guard = ClockEpochGuard(self.clock_settings.jump_limit_ms)
            self.clock_status = self.clock_monitor.snapshot()
            self._clock_log_reason = None
            self._clock_was_ready = False
            self.metrics_path = args.metrics_jsonl or ROOT / "runs/sb_vision/metrics" / (self.performance.session_id + ".jsonl")
            self.summary_path = args.metrics_summary or self.metrics_path.with_suffix(".summary.json")
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics = self.metrics_path.open("w")
            self._write_event({"record_type": "run_started", "context": self.performance.context,
                               "started_mono_ns": self.performance.started_ns})
            self._preloaded_model_load_s = None
            if args.preload_backend:
                factory = backend_factory or FoundationPoseBackend
                started = time.perf_counter()
                self.pipeline = PosePipeline(factory(profile, args), profile,
                                             detector_hz=args.detector_hz,
                                             miss_grace_s=args.miss_grace_seconds,
                                             reselect_after_s=args.reselect_after_seconds)
                self._preloaded_model_load_s = time.perf_counter() - started
                self.get_logger().info(f"YOLO/FoundationPose backend preloaded in {self._preloaded_model_load_s:.3f}s")
            prefix = "/" + args.camera_prefix.strip("/")
            rgb_topic, depth_topic = image_topics(prefix, args.image_transport)
            image_type = CompressedImage if args.image_transport == "compressed" else Image
            for key, topic, kind in (("rgb", rgb_topic, image_type), ("depth", depth_topic, image_type),
                                     ("rgb_info", prefix + "/color/camera_info", CameraInfo),
                                     ("depth_info", prefix + "/depth/camera_info", CameraInfo)):
                self.create_subscription(kind, topic, lambda msg, k=key: self.receive(k, msg), qos_profile_sensor_data)
            self.create_service(Trigger, "~/reset_target", self.reset_target)
            self.create_subscription(String, prefix + "/capture_status", self.camera_metadata, 10)
            self._tick_timer = self.create_timer(.005, self.tick)
            self._status_timer = self.create_timer(1.0, self.status)
            self.get_logger().info(f"ObjectPose output={args.object_pose_topic}; frame={args.frame_id}; "
                                   f"object={profile['object_id']}; images={args.image_transport}; "
                                   f"mesh dimensions={profile['expected_extents_m']} m")
            self.get_logger().info(f"Persistent sequence: {self.sequence_store.snapshot()}")
            self.get_logger().info(f"Events: {self.metrics_path}; summary: {self.summary_path}")

        def camera_metadata(self, msg):
            try:
                data = json.loads(msg.data)
                self.performance.context["camera"] = {key:str(data[key]) for key in
                    ("serial", "frame_id", "aligned_to", "image_transport") if key in data}
            except (ValueError, TypeError):
                pass

        def _write_event(self, record):
            if self.metrics is None:
                return
            record = dict(record, schema_version=2, session_id=self.performance.session_id)
            try:
                self.metrics.write(json.dumps(record, allow_nan=False) + "\n")
                self.metrics.flush()
            except OSError as exc:
                self.metrics_error = str(exc)
                self.get_logger().error(f"Metrics write failed: {exc}")

        def _queue_reset(self, reason, mono_ns=None):
            mono_ns = time.monotonic_ns() if mono_ns is None else mono_ns
            with self.reset_lock:
                self.pending_reset = (reason, mono_ns)
                self.reset_requested.set()
            self.performance.begin_recovery(reason, mono_ns)
            self._write_event({"record_type": "reset_requested", "reason": reason, "monotonic_ns": mono_ns})

        def _result_record(self, completed):
            result = completed.value
            frame = result.frame
            return {"stamp_ns": frame.stamp_ns, "depth_stamp_ns": frame.depth_stamp_ns,
                    "status": result.status, "mode": result.mode, "event_type": result.mode,
                    "operation": result.operation, "event_id": result.event_id,
                    "registration_attempt": result.registration_attempt,
                    "trigger_reason": result.trigger_reason, "trigger_mono_ns": result.trigger_mono_ns,
                    "full_search_requested": result.full_search_requested, "search": result.search,
                    "processing_start_mono_ns": result.processing_start_mono_ns,
                    "operation_start_mono_ns": result.operation_start_mono_ns,
                    "quality": result.quality, "timings": result.timings, "worker_s": completed.duration_s,
                    "observation_delay_s": (self.get_clock().now().nanoseconds-frame.stamp_ns)/1e9,
                    "target_generation": result.target_generation, "gpu_memory": result.gpu_memory,
                    "detection_fresh": result.detection_fresh, "detection_age_s": result.detection_age_s,
                    "estimate_valid": result.pose_center is not None, "error": result.error, "published": False}

        def _summary(self, final=False):
            summary = self.performance.snapshot(gpu=self.gpu_sampler.snapshot() if self.gpu_sampler else None, final=final)
            summary.update(counts=dict(getattr(self, "counts", {})), fatal_error=getattr(self, "fatal_error", None),
                           metrics_io_error=self.metrics_error)
            try:
                write_summary(self.summary_path, summary)
            except OSError as exc:
                self.metrics_error = str(exc)
                self.get_logger().error(f"Metrics summary write failed: {exc}")
            return summary

        def reset_target(self, request, response):
            self.reset_cutoff_ns = self.get_clock().now().nanoseconds
            self.latest = None
            self._queue_reset("manual_reset")
            response.success = True
            response.message = "Target reset requested; publication sequence is preserved"
            return response

        def infer(self, request):
            _, frame = request
            model_load_s = None
            if self.pipeline is None:
                factory = backend_factory or FoundationPoseBackend
                started = time.perf_counter()
                self.pipeline = PosePipeline(factory(profile, args), profile,
                                             detector_hz=args.detector_hz,
                                             miss_grace_s=args.miss_grace_seconds,
                                             reselect_after_s=args.reselect_after_seconds)
                model_load_s = time.perf_counter() - started
            with self.reset_lock:
                pending = self.pending_reset
                self.pending_reset = None
                self.reset_requested.clear()
            if pending is not None:
                self.pipeline.reset_target(reason=pending[0], trigger_mono_ns=pending[1])
            try:
                result = self.pipeline.process(frame)
            except PosePipelineError as exc:
                result = exc.result
            if model_load_s is not None:
                result.timings["model_load_s"] = model_load_s
            elif self._preloaded_model_load_s is not None:
                result.timings["model_load_s"] = self._preloaded_model_load_s
                self._preloaded_model_load_s = None
            return result

        def receive(self, key, msg):
            try:
                self.sync.add(key, msg)
                frame = self.sync.pop_latest()
                if frame is not None:
                    self.latest = frame  # overwrite backlog; retain original capture stamp
                    self.counts["paired_frames"] += 1
                    self.performance.context["image_size"] = list(frame.depth_m.shape[::-1])
            except (ValueError, TypeError) as exc:
                self.counts["invalid_rgbd"] += 1
                self.last_result = {"status": "invalid_rgbd", "detail": str(exc)}

        def tick(self):
            if self._closing or self.fatal_error:
                return
            self.clock_status = self.clock_monitor.snapshot()
            now_ns, mono_ns, uncertainty_ns = sample_clock_pair(lambda: self.get_clock().now().nanoseconds)
            decision = self.clock_guard.observe(self.clock_status["ready"], now_ns, mono_ns, uncertainty_ns)
            if decision.jumped:
                self.clock_monitor.invalidate("local_clock_jump")
                self.clock_status = dict(self.clock_status, ready=False, reason="local_clock_jump")
            if decision.changed:
                self.latest = None
                self.sync = RGBDSynchronizer(args.frame_id, args.max_rgbd_skew_ms)
                cause = "startup" if decision.ready and not self._clock_was_ready else (
                    "clock_recovered" if decision.ready else "clock_unavailable")
                self._queue_reset(cause)
                if decision.ready:
                    self._clock_was_ready = True
                    # Only use exposures captured after synchronization became usable.
                    self.reset_cutoff_ns = now_ns
            reason = self.clock_status.get("reason")
            if reason != self._clock_log_reason:
                self.get_logger().info(f"Clock sync: {reason}; ready={decision.ready}; server={self.clock_settings.server}")
                self._clock_log_reason = reason
            for completed in self.worker.poll():
                if completed.error:
                    self.counts["inference_error"] += 1
                    self.fatal_error = completed.error
                    self.get_logger().error(completed.error)
                    self._write_event({"record_type": "worker_error", "error": completed.error,
                                       "worker_s": completed.duration_s})
                    continue
                result = completed.value
                frame = result.frame
                now_ns = self.get_clock().now().nanoseconds
                delay = (now_ns - frame.stamp_ns) / 1e9
                record = self._result_record(completed)
                if result.error:
                    self.fatal_error = result.error
                    self.get_logger().error(result.error)
                clock_valid = decision.ready and completed.request[0] == decision.epoch
                if not clock_valid:
                    record["status"] = "clock_unsynchronized" if not decision.ready else "old_clock_epoch"
                elif result.pose_center is not None and frame.stamp_ns <= self.reset_cutoff_ns:
                    record["status"] = "before_reset_cutoff"
                if clock_valid and result.pose_center is not None and frame.stamp_ns > self.reset_cutoff_ns:
                    try:
                        sequence = self.gate.admit(frame, now_ns)
                    except (OSError, RuntimeError) as exc:
                        self.fatal_error = f"Sequence allocation failed; publishing stopped: {exc}"
                        self.counts["sequence_state_error"] += 1
                        self.get_logger().error(self.fatal_error)
                        return
                    if sequence is not None:
                        # A rare range reservation includes fsync. If storage was
                        # slow, consume the number but never publish a stale pose.
                        now_ns = self.get_clock().now().nanoseconds
                        delay_ns = now_ns - frame.stamp_ns
                        record["observation_delay_s"] = delay_ns / 1e9
                        if not -5000000 <= delay_ns <= self.gate.max_delay_ns:
                            record["status"] = "expired_during_sequence_allocation"
                            sequence = None
                    if sequence is not None:
                        msg = ObjectPose()
                        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(frame.stamp_ns, 1000000000)
                        msg.header.frame_id = frame.frame_id
                        msg.object_id = profile["object_id"]
                        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, result.pose_center[:3, 3])
                        q = quaternion_xyzw(result.pose_center[:3, :3])
                        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = map(float, q)
                        msg.confidence = float(result.confidence)
                        msg.covariance = result.covariance.reshape(-1).tolist()
                        msg.source = "foundationpose"
                        msg.sequence = sequence
                        self.pub.publish(msg)
                        record.update(published=True, sequence=sequence, published_mono_ns=time.monotonic_ns())
                        self.counts["published"] += 1
                    elif record["status"] != "expired_during_sequence_allocation":
                        record["status"] = "stale_future_or_duplicate_result"
                if not record["published"]:
                    self.counts[record["status"]] += 1
                self.latencies.append(record["observation_delay_s"])
                self.last_result = record
                self.performance.record(record, now_ns=record.get("published_mono_ns"))
                self._write_event(record)
            if self.performance.check_output_timeout(args.max_observation_delay):
                self._write_event({"record_type": "output_timeout", "recovery": self.performance.active_recovery})
            if self.fatal_error or self.worker.busy or self.latest is None:
                return
            frame, self.latest = self.latest, None
            if not decision.ready:
                self.counts["clock_unsynchronized_input"] += 1
                return
            age = (self.get_clock().now().nanoseconds - frame.stamp_ns) / 1e9
            if not -.005 <= age <= args.max_observation_delay or frame.stamp_ns <= self.reset_cutoff_ns:
                self.counts["stale_input"] += 1
                return
            self.worker.submit((decision.epoch, frame))

        def status(self):
            performance = self._summary()
            msg = String()
            msg.data = json.dumps({"counts": dict(self.counts), "last": self.last_result,
                "worker_busy": self.worker.busy, "sequence": self.gate.sequence,
                "delay_p95_s": float(np.percentile(self.latencies, 95)) if self.latencies else None,
                "clock_sync": self.clock_status, "clock_epoch": self.clock_guard.epoch,
                "sequence_state": self.sequence_store.snapshot(),
                "performance": performance,
                "fatal": bool(self.fatal_error)}, allow_nan=False)
            self.status_pub.publish(msg)

        def close(self):
            if self._closing:
                return
            self._closing = True
            for name in ("_tick_timer", "_status_timer"):
                timer = getattr(self, name, None)
                if timer is not None:
                    timer.cancel()
            try:
                if self.clock_monitor is not None:
                    self.clock_monitor.close()
                if self.worker is not None:
                    self.worker.close(wait=True)
                    for completed in self.worker.poll():
                        if completed.value is not None:
                            record = self._result_record(completed)
                            record["status"] = "shutdown_discard"
                            self.performance.record(record)
                            self._write_event(record)
                        elif completed.error:
                            self._write_event({"record_type": "worker_error_at_shutdown", "error": completed.error})
            finally:
                try:
                    if self.gpu_sampler is not None:
                        self.gpu_sampler.close()
                    if self.performance is not None and hasattr(self, "summary_path"):
                        self._summary(final=True)
                        self._write_event({"record_type": "run_ended", "summary_path": str(self.summary_path)})
                    if self.metrics:
                        self.metrics.close()
                finally:
                    if self.sequence_store is not None:
                        self.sequence_store.close()

    return VisionNode()


def main(argv=None, *, backend_factory=None):
    parser = build_parser()
    args, ros_args = parser.parse_known_args(argv)
    if args.max_observation_delay <= 0 or args.max_rgbd_skew_ms <= 0:
        parser.error("time limits must be positive")
    if args.detector_hz <= 0 or args.miss_grace_seconds <= 0 or args.reselect_after_seconds <= 0:
        parser.error("detector and target-loss time limits must be positive")
    if not np.isfinite(args.gpu_memory_sample_ms) or args.gpu_memory_sample_ms <= 0:
        parser.error("--gpu-memory-sample-ms must be positive and finite")
    profile = load_object_profile(args.object_profile)
    import rclpy
    rclpy.init(args=ros_args)
    node = None
    try:
        node = make_node(args, profile, backend_factory=backend_factory)
        started = time.monotonic()
        while rclpy.ok() and not node.fatal_error:
            rclpy.spin_once(node, timeout_sec=.02)
            if args.max_runtime_seconds and time.monotonic() - started >= args.max_runtime_seconds:
                break
        return 1 if node.fatal_error else 0
    except KeyboardInterrupt:
        return 0
    except (SequenceStateError, OSError, ValueError) as exc:
        print(f"Vision startup/runtime error: {exc}", file=sys.stderr)
        return 1
    finally:
        if node:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
