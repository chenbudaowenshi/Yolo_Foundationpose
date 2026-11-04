#!/usr/bin/env python3
"""Sweep the simulated head until YOLO detects the configured target object."""

import math
import os
from pathlib import Path
import sys
import threading
import time
import argparse


YOLO_PYTHON = "/home/robot/anaconda3/envs/pose/bin/python"


def _restart_with_yolo_python() -> None:
    """Use the existing Python 3.10 environment containing ROS 2 and YOLO."""
    if __name__ != "__main__" or not os.path.isfile(YOLO_PYTHON):
        return
    if os.path.realpath(sys.executable) == os.path.realpath(YOLO_PYTHON):
        return

    print(
        f"Switching from {sys.executable} to {YOLO_PYTHON} for ROS 2 + YOLO...",
        file=sys.stderr,
        flush=True,
    )
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    os.execve(
        YOLO_PYTHON,
        [YOLO_PYTHON, os.path.abspath(__file__), *sys.argv[1:]],
        environment,
    )


_restart_with_yolo_python()

try:
    import numpy as np
    import rclpy
    import torch
    from communication.msg import ActuatorCmds
    from rclpy.node import Node
    from rclpy.qos import (
        QoSProfile,
        ReliabilityPolicy,
        DurabilityPolicy,
        qos_profile_sensor_data,
    )
    from sensor_msgs.msg import Image
    from ultralytics import YOLO
    import cv2
except ImportError as error:
    raise SystemExit(
        "Cannot load ROS 2, YOLO, or the BXI messages. Run this first:\n"
        "  source /home/robot/FoundationPose/"
        "bxi_rl_controller_ros2_example/install/setup.bash\n"
        f"  {YOLO_PYTHON} /home/robot/FoundationPose/my_sim/head_control.py\n"
        f"Original error: {error}"
    ) from error


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
TOPIC_NAME = "/simulation/actuators_cmds_override"
CAMERA_TOPIC = "/simulation/head_depth_camera/color/image_raw"
YOLO_MODEL_PATH = REPOSITORY_ROOT / "ultralytics/ultralytics/yolov8n-seg.pt"
TARGET_CLASS_NAME = "bottle"
PUBLISH_RATE_HZ = 20.0
YAW_AMPLITUDE_RAD = 0.45
SWEEP_PERIOD_SEC = 4.0
PITCH_POSITION_RAD = 0.0
DETECTION_CONFIDENCE = 0.50
DETECTION_IMAGE_SIZE = 640
DETECTION_RATE_HZ = 10.0
REQUIRED_CONSECUTIVE_DETECTIONS = 2
CAMERA_WARNING_DELAY_SEC = 3.0
CAMERA_WARNING_PERIOD_SEC = 5.0
PREVIEW_WINDOW_NAME = "ELF3 head scan / bottle detection"


def image_message_to_bgr(message: Image) -> np.ndarray:
    """Decode a MuJoCo ``rgb8``/``bgr8`` ROS image without cv_bridge."""
    height = int(message.height)
    width = int(message.width)
    encoding = str(message.encoding).lower()
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image size: {width}x{height}")
    if encoding not in {"rgb8", "bgr8"}:
        raise ValueError(f"unsupported image encoding: {message.encoding!r}")

    row_bytes = width * 3
    step = int(message.step)
    if step < row_bytes:
        raise ValueError(f"image step is {step}, expected at least {row_bytes}")
    required_bytes = step * height
    if len(message.data) < required_bytes:
        raise ValueError(
            f"image contains {len(message.data)} bytes, expected {required_bytes}"
        )

    rows = np.frombuffer(message.data, dtype=np.uint8, count=required_bytes)
    pixels = rows.reshape(height, step)[:, :row_bytes].reshape(height, width, 3)
    if encoding == "rgb8":
        pixels = pixels[:, :, ::-1]
    return np.ascontiguousarray(pixels).copy()


class HeadControlPublisher(Node):
    """Publish a yaw sweep and latch its position after an object is detected."""

    def __init__(self, *, show_preview: bool = True, exit_on_detection: bool = False, detection_file: str | None = None, camera_topic: str = CAMERA_TOPIC) -> None:
        super().__init__("head_control_publisher")

        if not YOLO_MODEL_PATH.is_file():
            raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL_PATH}")

        self.camera_topic = str(camera_topic)
        self.publisher = self.create_publisher(
            ActuatorCmds,
            TOPIC_NAME,
            qos_profile_sensor_data,
        )
        self.image_subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )
        # MuJoCo package releases differ in the reliability offered by their
        # camera publisher.  Keep the low-latency sensor subscription above,
        # and add a reliable fallback so either DDS QoS variant is received.
        self.image_subscription_reliable = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )

        self.command = ActuatorCmds()
        self.command.actuators_name = ["head_z_joint", "head_y_joint"]
        # pos[0] = 左右 yaw (head_z_joint), pos[1] = 上下 pitch (head_y_joint)
        self.command.pos = [0.0, PITCH_POSITION_RAD]
        self.command.kp = [16.747, 16.747]
        self.command.kd = [1.066, 1.066]
        self.start_time = time.monotonic()
        self.scan_started_at = None
        self.last_camera_warning_at = float("-inf")
        self.camera_graph_reported = False
        self.last_camera_graph_check_at = float("-inf")
        self.last_image_error_at = float("-inf")
        self.received_image = False
        self.stop_logged = False
        self.failure_logged = False
        self.show_preview = bool(show_preview)
        self.exit_on_detection = bool(exit_on_detection)
        self.detection_file = Path(detection_file).expanduser() if detection_file else None
        self._shutdown_timer = None

        self.object_detected = threading.Event()
        self.vision_failed = threading.Event()
        self.worker_stop = threading.Event()
        self.model_ready = threading.Event()
        self.frame_condition = threading.Condition()
        self.latest_frame = None
        self.latest_frame_sequence = 0
        self.detection_info_lock = threading.Lock()
        self.detection_info = ("", 0.0)
        self.detection_box = None
        self.vision_error = ""

        self.detection_worker = threading.Thread(
            target=self.detection_loop,
            name="yolo_detection_worker",
            daemon=True,
        )
        self.detection_worker.start()

        self.timer = self.create_timer(
            1.0 / PUBLISH_RATE_HZ,
            self.publish_command,
        )
        self.get_logger().info(
            "Sweeping head left/right on "
            f"{TOPIC_NAME}: +/-{YAW_AMPLITUDE_RAD:.2f} rad, "
            f"period={SWEEP_PERIOD_SEC:.1f} s, rate={PUBLISH_RATE_HZ:g} Hz"
        )
        self.get_logger().info(
            f"YOLO input={self.camera_topic}, model={YOLO_MODEL_PATH}, "
            f"target={TARGET_CLASS_NAME!r}, confidence>={DETECTION_CONFIDENCE:.2f}"
        )
        self.get_logger().info(
            "Head scan starts immediately while YOLO warms up; the detected "
            "bottle pose will latch the current yaw."
        )

    def image_callback(self, message: Image) -> None:
        """Replace the pending image without ever blocking on YOLO inference."""
        if self.object_detected.is_set() or self.vision_failed.is_set():
            return

        try:
            frame = image_message_to_bgr(message)
        except Exception as error:
            now = time.monotonic()
            if now - self.last_image_error_at >= CAMERA_WARNING_PERIOD_SEC:
                self.get_logger().warning(f"Cannot decode camera image: {error}")
                self.last_image_error_at = now
            return

        # Keep a visible, low-latency preview in the same process as ROS.  This
        # makes it obvious that the camera is live while the head is sweeping.
        if self.show_preview:
            try:
                preview = frame.copy()
                with self.detection_info_lock:
                    box = None if self.detection_box is None else self.detection_box.copy()
                    label, confidence = self.detection_info
                if box is not None:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 0), 3)
                    cv2.putText(preview, f"{label} {confidence:.2f}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
                status = "LOCKED" if self.object_detected.is_set() else ("WARMING UP" if not self.model_ready.is_set() else "SCANNING")
                cv2.putText(preview, f"HEAD {status} yaw={self.command.pos[0]:+.2f} rad", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 180, 255), 2)
                cv2.imshow(PREVIEW_WINDOW_NAME, preview)
                cv2.waitKey(1)
            except Exception as error:
                self.show_preview = False
                self.get_logger().warning(f"preview disabled: {error}")

        if not self.received_image:
            self.received_image = True
            self.get_logger().info(
                f"First MuJoCo RGB frame received: {frame.shape[1]}x{frame.shape[0]}"
            )

        with self.frame_condition:
            self.latest_frame = frame
            self.latest_frame_sequence += 1
            self.frame_condition.notify()

    def detection_loop(self) -> None:
        """Run YOLO in one background thread using only the newest RGB frame."""
        try:
            device = 0 if torch.cuda.is_available() else "cpu"
            model = YOLO(str(YOLO_MODEL_PATH))
            names = (
                {int(key): str(value) for key, value in model.names.items()}
                if isinstance(model.names, dict)
                else {index: str(value) for index, value in enumerate(model.names)}
            )
            matching_classes = [
                class_id
                for class_id, name in names.items()
                if name.casefold() == TARGET_CLASS_NAME.casefold()
            ]
            if len(matching_classes) != 1:
                raise ValueError(
                    f"YOLO class {TARGET_CLASS_NAME!r} is not unique; available={names}"
                )
            target_class_id = matching_classes[0]

            # Warm-up happens outside the ROS callback thread, so head commands
            # continue to be published while Torch initializes its kernels.
            warmup_image = np.zeros(
                (DETECTION_IMAGE_SIZE, DETECTION_IMAGE_SIZE, 3),
                dtype=np.uint8,
            )
            model.predict(
                warmup_image,
                conf=DETECTION_CONFIDENCE,
                imgsz=DETECTION_IMAGE_SIZE,
                device=device,
                verbose=False,
            )
            self.model_ready.set()
            self.get_logger().info(f"YOLO ready on device={device}")

            processed_sequence = 0
            positive_count = 0
            minimum_interval = 1.0 / DETECTION_RATE_HZ
            last_inference_at = float("-inf")

            while not self.worker_stop.is_set() and not self.object_detected.is_set():
                with self.frame_condition:
                    new_frame_ready = self.frame_condition.wait_for(
                        lambda: (
                            self.worker_stop.is_set()
                            or self.latest_frame_sequence != processed_sequence
                        ),
                        timeout=0.5,
                    )
                    if self.worker_stop.is_set():
                        return
                    if not new_frame_ready:
                        continue

                remaining = minimum_interval - (time.monotonic() - last_inference_at)
                if remaining > 0.0 and self.worker_stop.wait(remaining):
                    return

                with self.frame_condition:
                    if self.latest_frame is None:
                        continue
                    frame = self.latest_frame
                    processed_sequence = self.latest_frame_sequence

                result = model.predict(
                    frame,
                    conf=DETECTION_CONFIDENCE,
                    imgsz=DETECTION_IMAGE_SIZE,
                    device=device,
                    verbose=False,
                )[0]
                last_inference_at = time.monotonic()
                boxes = result.boxes

                if boxes is None or len(boxes) == 0:
                    positive_count = 0
                    continue

                class_ids = boxes.cls.to(dtype=torch.int64)
                candidate_indices = torch.nonzero(
                    class_ids == target_class_id,
                    as_tuple=False,
                ).flatten()
                if candidate_indices.numel() == 0:
                    positive_count = 0
                    continue

                confidences = boxes.conf
                candidate_confidences = confidences[candidate_indices]
                best_index = int(
                    candidate_indices[int(candidate_confidences.argmax().item())].item()
                )
                confidence = float(confidences[best_index].item())
                class_id = int(boxes.cls[best_index].item())
                label = names[class_id]

                positive_count += 1
                if positive_count < REQUIRED_CONSECUTIVE_DETECTIONS:
                    continue

                with self.detection_info_lock:
                    self.detection_info = (label, confidence)
                    self.detection_box = (
                        boxes.xyxy[best_index].detach().cpu().numpy().astype(float)
                    )
                self.object_detected.set()
                with self.frame_condition:
                    self.frame_condition.notify_all()
                return
        except Exception as error:
            with self.detection_info_lock:
                self.vision_error = f"{type(error).__name__}: {error}"
            self.vision_failed.set()
            with self.frame_condition:
                self.frame_condition.notify_all()

    def publish_command(self) -> None:
        now = time.monotonic()
        self.command.header.stamp = self.get_clock().now().to_msg()

        if self.object_detected.is_set():
            if not self.stop_logged:
                with self.detection_info_lock:
                    label, confidence = self.detection_info
                    box = self.detection_box
                self.get_logger().info(
                    f"Object detected: {label} ({confidence:.3f}); "
                    f"box={None if box is None else box.tolist()}; "
                    f"holding commanded head yaw at {self.command.pos[0]:.3f} rad "
                    "(the pose node will read the measured joint angle)"
                )
                self.stop_logged = True
                if self.detection_file is not None:
                    self.detection_file.parent.mkdir(parents=True, exist_ok=True)
                    self.detection_file.write_text("detected\n", encoding="utf-8")
                if self.exit_on_detection and self._shutdown_timer is None:
                    self._shutdown_timer = self.create_timer(
                        0.6, self._shutdown_after_detection
                    )
        elif self.vision_failed.is_set():
            if not self.failure_logged:
                with self.detection_info_lock:
                    detail = self.vision_error
                self.get_logger().error(
                    f"YOLO failed; holding the current head position: {detail}"
                )
                self.failure_logged = True
        else:
            # Start sweeping immediately.  YOLO warms up in the background;
            # keeping the head moving during that time lets the camera cover
            # the scene instead of appearing frozen at zero yaw.
            if self.scan_started_at is None:
                self.scan_started_at = now
            elapsed = now - self.scan_started_at
            phase = 2.0 * math.pi * elapsed / SWEEP_PERIOD_SEC
            self.command.pos[0] = YAW_AMPLITUDE_RAD * math.sin(phase)

            if (
                not self.received_image
                and now - self.start_time >= CAMERA_WARNING_DELAY_SEC
                and now - self.last_camera_warning_at >= CAMERA_WARNING_PERIOD_SEC
            ):
                # DDS discovery may complete well after node creation.  Recheck
                # on every warning interval instead of caching an early zero.
                if now - self.last_camera_graph_check_at >= CAMERA_WARNING_PERIOD_SEC:
                    try:
                        publishers = self.get_publishers_info_by_topic(self.camera_topic)
                        qos = ", ".join(
                            str(getattr(info.qos_profile, "reliability", "?"))
                            for info in publishers
                        ) or "none"
                        topic_types = dict(self.get_topic_names_and_types()).get(
                            self.camera_topic, []
                        )
                        self.get_logger().warning(
                            "camera graph: publishers={} reliability={} types={} env="
                            "ROS_DOMAIN_ID={}, RMW_IMPLEMENTATION={}, "
                            "ROS_LOCALHOST_ONLY={}"
                            .format(
                                len(publishers),
                                qos,
                                topic_types or "none",
                                os.environ.get("ROS_DOMAIN_ID", "<unset>"),
                                os.environ.get("RMW_IMPLEMENTATION", "<unset>"),
                                os.environ.get("ROS_LOCALHOST_ONLY", "<unset>"),
                            )
                        )
                    except Exception as error:
                        self.get_logger().warning(
                            f"camera graph introspection failed: {error}"
                        )
                    self.camera_graph_reported = True
                    self.last_camera_graph_check_at = now
                self.get_logger().warning(
                    f"Waiting for MuJoCo camera images on {self.camera_topic}"
                )
                self.last_camera_warning_at = now

        self.command.pos[1] = PITCH_POSITION_RAD
        self.publisher.publish(self.command)

    def _shutdown_after_detection(self) -> None:
        if self._shutdown_timer is not None:
            self._shutdown_timer.cancel()
        rclpy.shutdown()

    def close(self) -> None:
        """Stop the YOLO worker without leaving inference work queued."""
        self.worker_stop.set()
        with self.frame_condition:
            self.frame_condition.notify_all()
        if self.detection_worker.is_alive():
            self.detection_worker.join(timeout=3.0)
        if self.show_preview:
            try:
                cv2.destroyWindow(PREVIEW_WINDOW_NAME)
            except Exception:
                pass


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description="Sweep ELF3 head and detect bottle")
    parser.add_argument("--show", dest="show_preview", action=argparse.BooleanOptionalAction, default=True,
                        help="show live camera/detection window (default: on)")
    parser.add_argument("--exit-on-detection", action="store_true",
                        help="stop after a stable target detection")
    parser.add_argument("--detection-file", default=None,
                        help="write this marker after stable target detection")
    parser.add_argument("--camera-topic", default=CAMERA_TOPIC,
                        help="RGB image topic published by MuJoCo")
    parsed, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args)
    node = HeadControlPublisher(show_preview=parsed.show_preview,
                                exit_on_detection=parsed.exit_on_detection,
                                detection_file=parsed.detection_file,
                                camera_topic=parsed.camera_topic)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
