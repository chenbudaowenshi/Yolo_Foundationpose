"""Capture-time IMU alignment shared by ROS2 and local RealSense perception."""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

from camera_contract import capture_stamp_ns
from upright_pose_utils import GravityAxisFilter


@dataclass(frozen=True)
class ImuMatch:
    ready: bool
    reason: str
    image_stamp_ns: int | None
    imu_stamp_ns: int | None = None
    source_frame: str | None = None
    camera_frame: str | None = None
    age_s: float | None = None
    skew_ms: float | None = None
    gravity: object = None

    @property
    def axis_camera(self):
        return self.gravity.axis_camera if self.ready else None

    def metadata(self):
        return {"required": True, "ready": self.ready, "reason": self.reason,
                "image_stamp_ns": self.image_stamp_ns, "imu_stamp_ns": self.imu_stamp_ns,
                "source_frame": self.source_frame, "camera_frame": self.camera_frame,
                "age_s": self.age_s, "skew_ms": self.skew_ms}


class ImuAlignment:
    def __init__(self, *, maximum_skew_ms=30., timeout_s=.5, smoothing=.1, capacity=128):
        if not np.isfinite([maximum_skew_ms, timeout_s]).all() or min(maximum_skew_ms, timeout_s) <= 0:
            raise ValueError("IMU skew/age limits must be positive and finite")
        self.maximum_skew_ms = maximum_skew_ms
        self.timeout_s = timeout_s
        self.smoothing = smoothing
        self.capacity = capacity
        self.samples = {}
        self.filter = GravityAxisFilter(smoothing=smoothing)
        self.filtered_stamp = 0
        self.filtered_frame = None
        self.cached = {}
        self.last_error = "waiting_for_imu"

    def add(self, stamp_ns, frame_id, acceleration, *, now_ns=None):
        now_ns = time.time_ns() if now_ns is None else now_ns
        if type(stamp_ns) is not int or stamp_ns <= 0:
            self.samples.clear()
            self.last_error = "invalid_imu_timestamp"
            return False
        if stamp_ns > now_ns + 5_000_000:
            self.samples.clear()
            self.last_error = "imu_timestamp_in_future"
            return False  # Never retain future-dated data until it appears current.
        if stamp_ns in self.samples:
            return False  # Replayed samples do not refresh timestamps or smoothing.
        vector = np.asarray(acceleration, float).reshape(-1)
        reason = "ok"
        if not frame_id:
            reason = "empty_imu_frame"
        elif vector.shape != (3,) or not np.isfinite(vector).all():
            reason = "invalid_imu_acceleration"
        elif not 7.0 <= np.linalg.norm(vector) <= 12.5:
            reason = "invalid_gravity_norm"
        self.samples[stamp_ns] = (frame_id, vector.copy(), reason)
        while len(self.samples) > self.capacity:
            self.samples.pop(min(self.samples))
        self.last_error = reason
        return reason == "ok"

    def match(self, image_stamp_ns, camera_frame, *, now_ns=None, rotate=None):
        now_ns = time.time_ns() if now_ns is None else now_ns
        if image_stamp_ns is None or image_stamp_ns <= 0 or not camera_frame:
            return ImuMatch(False, "waiting_for_image_timestamp", image_stamp_ns)
        if not self.samples:
            return ImuMatch(False, self.last_error, image_stamp_ns, camera_frame=camera_frame)
        stamp = min(self.samples, key=lambda s: (abs(s-image_stamp_ns), s))
        source, vector, error = self.samples[stamp]
        age, skew = (now_ns-stamp)/1e9, (stamp-image_stamp_ns)/1e6
        fields = dict(image_stamp_ns=image_stamp_ns, imu_stamp_ns=stamp, source_frame=source,
                      camera_frame=camera_frame, age_s=age, skew_ms=skew)
        if age < -.005:
            error = "imu_timestamp_in_future"
        elif age > self.timeout_s:
            error = "imu_stale"
        elif abs(skew) > self.maximum_skew_ms:
            error = "imu_image_time_mismatch"
        if error != "ok":
            return ImuMatch(False, error, **fields)
        try:
            if source != camera_frame:
                if rotate is None:
                    raise ValueError("no TF provider")
                vector = rotate(vector, source, camera_frame, stamp)
            if not np.isfinite(vector).all() or not 7.0 <= np.linalg.norm(vector) <= 12.5:
                raise ValueError("invalid rotated acceleration")
        except Exception as exc:
            return ImuMatch(False, "imu_tf_unavailable: " + str(exc), **fields)
        if camera_frame != self.filtered_frame:
            self.filter = GravityAxisFilter(smoothing=self.smoothing)
            self.filtered_stamp = 0
            self.filtered_frame = camera_frame
            self.cached.clear()
        if stamp > self.filtered_stamp:
            gravity = self.filter.update(vector)
            self.cached[stamp] = gravity
            self.filtered_stamp = stamp
            while len(self.cached) > self.capacity:
                self.cached.pop(min(self.cached))
        gravity = self.cached.get(stamp)
        if gravity is None or gravity.axis_camera is None:
            return ImuMatch(False, "imu_filter_time_reversal", **fields)
        return ImuMatch(True, "ok", gravity=gravity, **fields)


def sdk_stamp(frame):
    """Use the same epoch/exposure convention as the RGB-D camera publisher."""
    import pyrealsense2 as rs
    if not frame:
        raise ValueError("missing SDK frame")
    metadata = {}
    for name in ("frame_timestamp", "sensor_timestamp"):
        key = getattr(rs.frame_metadata_value, name)
        if frame.supports_frame_metadata(key):
            metadata[name] = frame.get_frame_metadata(key)
    return capture_stamp_ns(frame.get_timestamp(), frame.get_frame_timestamp_domain(), metadata)[0]


def add_sdk_acceleration(alignment, frames, rotation, camera_frame):
    import pyrealsense2 as rs
    frame = frames.first_or_default(rs.stream.accel)
    if not frame:
        return
    try:
        value = frame.as_motion_frame().get_motion_data()
        vector = np.array([value.x, value.y, value.z])
        R = np.asarray(rotation.rotation).reshape(3, 3, order="F")
        alignment.add(sdk_stamp(frame), camera_frame, R @ vector)
    except ValueError:
        alignment.last_error = "invalid_sdk_imu_clock"


class RosImuInput:
    def __init__(self, node, args):
        from sensor_msgs.msg import Imu
        from rclpy.qos import qos_profile_sensor_data
        self.node, self.args = node, args
        self.alignment = ImuAlignment(maximum_skew_ms=args.max_imu_skew_ms,
                                      timeout_s=args.imu_timeout, smoothing=args.gravity_smoothing)
        self.count = 0
        self.tf_buffer = self.tf_listener = None
        self.subscription = node.create_subscription(Imu, args.imu_topic, self.receive, qos_profile_sensor_data)

    def receive(self, msg):
        self.count += 1
        stamp = int(msg.header.stamp.sec)*1_000_000_000 + int(msg.header.stamp.nanosec)
        source = self.args.imu_frame or msg.header.frame_id
        a = msg.linear_acceleration
        self.alignment.add(stamp, source.strip().lstrip('/'), [a.x, a.y, a.z])

    def rotate(self, vector, source, target, stamp_ns):
        from rclpy.time import Time
        from tf2_ros import Buffer, TransformListener
        from scipy.spatial.transform import Rotation
        if self.tf_buffer is None:
            self.tf_buffer = Buffer(node=self.node)
            self.tf_listener = TransformListener(self.tf_buffer, self.node, spin_thread=False)
        transform = self.tf_buffer.lookup_transform(target, source, Time(nanoseconds=stamp_ns))
        q = transform.transform.rotation
        return Rotation.from_quat([q.x, q.y, q.z, q.w]).apply(vector)

    def match(self, frame, *, now_ns=None):
        return self.alignment.match(frame.stamp_ns if frame is not None else None,
                                    frame.frame_id if frame is not None else None,
                                    now_ns=now_ns, rotate=self.rotate)
