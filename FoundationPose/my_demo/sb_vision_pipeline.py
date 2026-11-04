"""Measured FoundationPose register/track pipeline for the grasp interface."""
from __future__ import annotations

from dataclasses import dataclass, field
import time
import traceback
import numpy as np

from sb_pose_contract import assess_quality, box_iou, profile_target_class, rigid_transform, resize_pose_inputs


@dataclass
class PoseResult:
    frame: object
    pose_center: np.ndarray | None = None
    confidence: float = 0.0
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    status: str = "no_detection"
    mode: str = "none"
    target_generation: int = 0
    timings: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    operation: str = "none"
    event_id: int = 0
    registration_attempt: int | None = None
    trigger_reason: str | None = None
    trigger_mono_ns: int | None = None
    full_search_requested: bool | None = None
    search: dict = field(default_factory=dict)
    processing_start_mono_ns: int = 0
    operation_start_mono_ns: int | None = None
    gpu_memory: dict | None = None
    error: str | None = None
    detection_fresh: bool = False
    detection_age_s: float | None = None


class PosePipelineError(RuntimeError):
    def __init__(self, result):
        super().__init__(result.error)
        self.result = result


class PosePipeline:
    """Single worker owns the GPU context. Selection never changes the measured pose."""
    def __init__(self, backend, profile, reselect_after_s=1.0,
                 detector_hz=10.0, miss_grace_s=0.30):
        self.backend = backend
        self.center = rigid_transform(profile["T_model_object_center"])
        self.bounds = np.asarray(backend.bounds, float)
        self.last_box = None
        self.last_seen_ns = 0
        self.last_processed_ns = 0
        self.last_mode = "none"
        self.has_pose = False
        self.force_full = False
        self.generation = 0
        self.last_valid_ns = 0
        self.quality_failures = 0
        self.reselect_after_ns = int(reselect_after_s * 1e9)
        self.event_id = 0
        self.registration_attempts = 0
        self.pending_reason = "startup"
        self.pending_trigger_ns = time.monotonic_ns()
        if detector_hz <= 0:
            raise ValueError("detector_hz must be positive")
        if miss_grace_s <= 0:
            raise ValueError("miss_grace_s must be positive")
        self.detector_period_ns = int(1e9 / float(detector_hz))
        self.miss_grace_ns = int(float(miss_grace_s) * 1e9)
        self.last_detection_run_ns = 0
        self.selected_detection = None

    def reset_pose(self, reason="pose_reset", trigger_mono_ns=None):
        if self.has_pose or self.pending_reason is None:
            self.pending_reason = reason
            self.pending_trigger_ns = trigger_mono_ns or time.monotonic_ns()
        self.backend.reset()
        self.has_pose = False

    def reset_target(self, reason="manual_reset", trigger_mono_ns=None):
        self.reset_pose(reason, trigger_mono_ns)
        self.pending_reason = reason
        self.pending_trigger_ns = trigger_mono_ns or time.monotonic_ns()
        self.last_box = None
        self.last_seen_ns = 0
        self.force_full = False
        self.last_valid_ns = 0
        self.quality_failures = 0
        self.last_detection_run_ns = 0
        self.selected_detection = None

    def select(self, detections, frame):
        if self.last_box is not None:
            ranked = sorted(((box_iou(d.box, self.last_box), i, d) for i, d in enumerate(detections)), reverse=True)
            if ranked and ranked[0][0] >= .15:
                if len(ranked) > 1 and ranked[1][0] >= .15 and ranked[0][0] - ranked[1][0] < .10:
                    return None  # ambiguous overlap: do not jump between bottles
                return ranked[0][2]
            if frame.stamp_ns - self.last_seen_ns <= self.reselect_after_ns:
                return None
            self.reset_target("target_reselected")
        usable = []
        for index, detection in enumerate(detections):
            values = frame.depth_m[np.asarray(detection.mask, bool)]
            values = values[np.isfinite(values) & (values > .05)]
            if len(values) >= 16:
                usable.append((float(np.median(values)), index, detection))
        if not usable:
            return None
        self.generation += 1
        return min(usable, key=lambda item: (item[0], item[1]))[2]

    def process(self, frame):
        self.event_id += 1
        result = PoseResult(frame, event_id=self.event_id, processing_start_mono_ns=time.monotonic_ns())
        try:
            return self._process(frame, result)
        except Exception:
            result.error = traceback.format_exc()
            result.status = "inference_error"
            self.reset_pose("inference_exception")
            raise PosePipelineError(result)
        finally:
            result.timings["pipeline_s"] = (time.monotonic_ns()-result.processing_start_mono_ns)/1e9
            if hasattr(self.backend, "memory_snapshot"):
                try:
                    result.gpu_memory = self.backend.memory_snapshot()
                except Exception as exc:
                    result.gpu_memory = {"available": False, "error": str(exc)}

    def _process(self, frame, result):
        frame.validate()
        if frame.stamp_ns <= self.last_processed_ns:
            result.status = "duplicate_or_backwards_frame"
            return result
        previous = self.last_processed_ns
        self.last_processed_ns = frame.stamp_ns
        started = time.perf_counter()
        run_detection = (
            not self.has_pose
            or self.last_detection_run_ns == 0
            or frame.stamp_ns - self.last_detection_run_ns >= self.detector_period_ns
        )
        if run_detection:
            detections = self.backend.detect(frame)
            self.last_detection_run_ns = frame.stamp_ns
            result.timings["yolo_s"] = time.perf_counter() - started
            result.detection_fresh = True
            detection = self.select(detections, frame)
            if detection is not None:
                self.selected_detection = detection
        else:
            # Keep the detector at a lower rate while FoundationPose tracks
            # every RGB-D frame. A cached detection is only a short-lived
            # association seed; a new YOLO pass is still required to select a
            # different bottle or recover after the miss grace period.
            detections = []
            detection = self.selected_detection
            if detection is not None and frame.stamp_ns - self.last_seen_ns > self.miss_grace_ns:
                detection = None
            result.detection_fresh = False
        if self.last_detection_run_ns:
            result.detection_age_s = max(0.0, (frame.stamp_ns - self.last_detection_run_ns) / 1e9)
        result.target_generation = self.generation
        if detection is None:
            # Never publish an unobserved target. A short missed detection need
            # not discard a good internal seed and force expensive registration.
            if frame.stamp_ns - self.last_valid_ns > self.miss_grace_ns:
                self.reset_pose("target_not_observed")
            if detections:
                result.status = "no_target_match_or_depth"
                result.quality = {"detections": len(detections), "detector_confidence": [float(d.confidence) for d in detections]}
            return result
        if result.detection_fresh:
            self.last_box = np.asarray(detection.box).copy()
            self.last_seen_ns = frame.stamp_ns
        # A long initial registration can initialize the tracker but never gets
        # a fresh output timestamp. One new frame then refines that initial pose.
        if previous and frame.stamp_ns - previous > 500000000 and self.last_mode != "register":
            # A camera transport gap does not imply that the object moved.
            # Keep FoundationPose's last seed and try one local track update on
            # the first fresh frame. Quality checks below still force a clean
            # registration if the camera/object really changed.
            self.pending_reason = "frame_gap"
            self.pending_trigger_ns = time.monotonic_ns()
            self.quality_failures = 0
        mode = "track" if self.has_pose else "register"
        result.operation = mode
        if mode == "register":
            self.registration_attempts += 1
            result.mode = "initial_registration" if self.registration_attempts == 1 else "reregistration"
            result.registration_attempt = self.registration_attempts
            result.trigger_reason = self.pending_reason or "registration_retry"
            result.trigger_mono_ns = self.pending_trigger_ns
            result.full_search_requested = bool(self.force_full)
        else:
            result.mode = "tracking"
            result.trigger_reason = self.pending_reason or "continuous_tracking"
        checkpoint = self.backend.checkpoint() if self.has_pose and hasattr(self.backend, "checkpoint") else None
        result.operation_start_mono_ns = time.monotonic_ns()
        started = time.perf_counter()
        try:
            try:
                pose = self.backend.track(frame) if self.has_pose else self.backend.register(frame, detection, self.force_full)
            finally:
                result.timings["pose_s"] = time.perf_counter() - started
                result.timings[mode + "_s"] = result.timings["pose_s"]
                if mode == "register" and hasattr(self.backend, "registration_details"):
                    result.search = self.backend.registration_details()
                    result.full_search_requested = result.search.get("force_full_requested", result.full_search_requested)
            quality_started = time.perf_counter()
            pose = rigid_transform(pose)
            valid, confidence, quality = assess_quality(frame, detection, pose, self.bounds)
            result.timings["quality_s"] = time.perf_counter() - quality_started
        except Exception:
            raise
        self.last_mode = mode
        result.quality = quality
        if not valid:
            self.quality_failures += 1
            if mode == "track" and checkpoint is not None and self.quality_failures < 3:
                self.backend.restore(checkpoint)
            else:
                self.reset_pose("tracking_quality_failure" if mode == "track" else "registration_quality_rejected")
                self.pending_reason = "tracking_quality_failure" if mode == "track" else "registration_quality_rejected"
                self.pending_trigger_ns = time.monotonic_ns()
                self.force_full = True
            result.status = quality["reason"]
            return result
        self.has_pose = True
        self.last_valid_ns = frame.stamp_ns
        self.quality_failures = 0
        self.force_full = False
        self.pending_reason = None
        self.pending_trigger_ns = None
        result.pose_center = rigid_transform(pose @ self.center)
        result.confidence = confidence
        result.status = "valid"
        # covariance=0 explicitly means unknown. A sample covariance from a
        # smoothed tracker is not the uncertainty of this measured FP pose.
        return result


class FoundationPoseBackend:
    """Loaded inside the inference thread so the CUDA rasterizer has one owner."""
    def __init__(self, profile, args):
        import logging
        import os
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        os.environ["D435I_FOPOSE_REEXEC"] = "1"
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "mycpp/build"))
        import torch
        import trimesh
        import nvdiffrast.torch as dr
        import d435i_foundationpose as helpers
        from ultralytics import YOLO
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
        from learning.training import predict_pose_refine, predict_score
        if not torch.cuda.is_available():
            raise RuntimeError("FoundationPose requires a visible CUDA GPU on the vision computer")
        torch.cuda.set_device(int(args.device))
        self.torch = torch
        self.cuda_device = int(args.device)
        torch.cuda.reset_peak_memory_stats(self.cuda_device)
        helpers.install_foundationpose_opencv_compat(predict_pose_refine, predict_score)
        self.helpers = helpers
        self.args = helpers.build_parser().parse_args([])
        self.args.target_class = profile_target_class(profile, args.target_class)
        self.args.device = args.device
        self.args.mask_source = "auto"
        self.args.conf = args.detector_confidence
        self.args.imgsz = args.detector_image_size
        self.model = YOLO(str(args.yolo_model))
        if self.args.target_class not in self.model.names:
            raise ValueError(f"target class absent from detector: {self.model.names}")
        if "class_id" in profile and self.model.names[self.args.target_class] != profile["object_id"]:
            raise ValueError(f"YOLO 类别 {self.args.target_class} 是 {self.model.names[self.args.target_class]!r}，"
                             f"不是物体配置里的 {profile['object_id']!r}；请换用 5 类分割模型")
        mesh = helpers.load_mesh(Path(profile["mesh_path"]), trimesh)
        if not np.allclose(mesh.extents, profile["expected_extents_m"], rtol=0, atol=.0001):
            raise ValueError("mesh dimensions do not match object profile")
        self.bounds = mesh.bounds.copy()
        self.estimator = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
            mesh=mesh, scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
            glctx=dr.RasterizeCudaContext(), debug=0,
            debug_dir=str(root / "runs/sb_vision/foundationpose_debug"),
            registration_views=args.register_views, registration_inplane_step=90,
            registration_topk=args.register_topk)
        logging.getLogger().setLevel(logging.WARNING)
        self.register_iterations = args.register_iterations
        self.track_iterations = args.track_iterations
        self.pose_max_side = args.pose_max_side

    def detect(self, frame):
        # find_targets accepts metric floating-point depth with a unit scale.
        return self.helpers.find_targets(self.model, frame.color_bgr, frame.depth_m, 1.0, self.args)[0]

    def register(self, frame, detection, force_full=False):
        frame, mask = resize_pose_inputs(frame, self.pose_max_side, detection.mask)
        return self.estimator.register(K=frame.K, rgb=frame.color_bgr[:, :, ::-1].copy(),
            depth=frame.depth_m, ob_mask=mask, iteration=self.register_iterations,
            force_full=force_full)

    def track(self, frame):
        frame, _ = resize_pose_inputs(frame, self.pose_max_side)
        return self.estimator.track_one(K=frame.K, rgb=frame.color_bgr[:, :, ::-1].copy(),
            depth=frame.depth_m, iteration=self.track_iterations)

    def reset(self):
        self.estimator.pose_last = None

    def checkpoint(self):
        return self.estimator.pose_last.clone() if self.estimator.pose_last is not None else None

    def restore(self, state):
        self.estimator.pose_last = state

    def registration_details(self):
        return dict(getattr(self.estimator, "last_registration_metrics", {}))

    def memory_snapshot(self):
        torch, device = self.torch, self.cuda_device
        return {"source": "PyTorch CUDA allocator since backend initialization", "available": True,
                "device": device, "device_name": torch.cuda.get_device_name(device),
                "allocated_bytes": torch.cuda.memory_allocated(device),
                "reserved_bytes": torch.cuda.memory_reserved(device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
