"""RealSense RGB-aligned depth capture with SDK calibration and capture time."""
from __future__ import annotations

import time
import queue
import threading
import cv2
import numpy as np
import pyrealsense2 as rs

from camera_contract import capture_stamp_ns, ros_distortion, aligned_imu_samples


class RealSenseCamera:
    def __init__(self, width=1280, height=720, fps=30, timeout_ms=1000,
                 serial=None, enable_imu=False, accel_fps=100, gyro_fps=200):
        self.timeout_ms = int(timeout_ms)
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if serial:
            self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        if enable_imu:
            fmt = getattr(rs.format, "motion_xyz32f", None)
            if fmt is None:
                fmt = rs.format.motion_xyz
            self.config.enable_stream(rs.stream.accel, fmt, accel_fps)
            self.config.enable_stream(rs.stream.gyro, fmt, gyro_fps)
        device = self.config.resolve(rs.pipeline_wrapper(self.pipeline)).get_device()
        for sensor in device.query_sensors():
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled, 1)
            if sensor.supports(rs.option.frames_queue_size):
                sensor.set_option(rs.option.frames_queue_size, 1)
        self._stopped = False
        self.profile = self.pipeline.start(self.config)
        try:
            self.align = rs.align(rs.stream.color)
            color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_profile.get_intrinsics()
            self.fx, self.fy, self.cx, self.cy = intr.fx, intr.fy, intr.ppx, intr.ppy
            self.width, self.height = intr.width, intr.height
            self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=float)
            self.distortion_model, self.D = ros_distortion(intr.model, intr.coeffs)
            self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
            self.serial = self.profile.get_device().get_info(rs.camera_info.serial_number)
            self.sdk_distortion_model = str(intr.model)
            self._motion_rotations = {}
            if enable_imu:
                for stream in (rs.stream.accel, rs.stream.gyro):
                    extr = self.profile.get_stream(stream).get_extrinsics_to(color_profile)
                    # SDK rs2_extrinsics.rotation is column-major.
                    self._motion_rotations[stream] = np.asarray(extr.rotation).reshape(3, 3, order="F")
            self._started = time.monotonic()
            self._capture_thread = None
            self._capture_stop = threading.Event()
            self._capture_error = None
            self._raw_frames = queue.Queue(maxsize=1)
            self._motion_lock = threading.Lock()
            self._motion_history = {"accel": {}, "gyro": {}}
            if enable_imu:
                # Keep the native 30 Hz SDK samples even when ROS publishes at
                # 15 Hz. The previous SDK sample is often closest to exposure
                # midpoint; reading only in the ROS timer loses that sample.
                self._capture_thread = threading.Thread(target=self._capture_loop, name="realsense-capture", daemon=True)
                self._capture_thread.start()
            print(f"RealSense {self.serial}: {self.width}x{self.height}@{fps}; "
                  f"depth_scale={self.depth_scale}; D={self.D}; SDK={self.sdk_distortion_model}", flush=True)
        except BaseException:
            self.stop()
            raise

    @staticmethod
    def frame_clock(frame):
        metadata = {}
        for name in ("frame_timestamp", "sensor_timestamp", "actual_exposure", "frame_counter", "actual_fps"):
            key = getattr(rs.frame_metadata_value, name)
            if frame.supports_frame_metadata(key):
                metadata[name] = frame.get_frame_metadata(key)
        domain = str(frame.get_frame_timestamp_domain()).split(".")[-1]
        stamp, basis = capture_stamp_ns(frame.get_timestamp(), domain, metadata)
        return stamp, {"timestamp_domain": domain, "timestamp_basis": basis, **metadata}

    def _capture_loop(self):
        last_video_frame = None
        try:
            while not self._capture_stop.is_set():
                raw = self.pipeline.wait_for_frames(self.timeout_ms)
                for stream, rotation in self._motion_rotations.items():
                    sample = raw.first_or_default(stream)
                    if not sample:
                        continue
                    try:
                        stamp = self.frame_clock(sample)[0]
                    except ValueError:
                        continue  # SDK global clock may still be warming up.
                    value = sample.as_motion_frame().get_motion_data()
                    key = "accel" if stream == rs.stream.accel else "gyro"
                    with self._motion_lock:
                        history = self._motion_history[key]
                        history[stamp] = rotation @ np.array([value.x, value.y, value.z])
                        while len(history) > 128:
                            history.pop(min(history))
                color, depth = raw.get_color_frame(), raw.get_depth_frame()
                if not color or not depth or color.get_frame_number() == last_video_frame:
                    continue
                last_video_frame = color.get_frame_number()
                try:
                    self._raw_frames.put_nowait(raw)
                except queue.Full:
                    try:
                        self._raw_frames.get_nowait()
                    except queue.Empty:
                        pass
                    self._raw_frames.put_nowait(raw)
        except Exception as error:
            if not self._capture_stop.is_set():
                self._capture_error = error

    def get_frame(self):
        started = time.perf_counter()
        if self._capture_thread is not None:
            try:
                raw = self._raw_frames.get(timeout=self.timeout_ms / 1000.)
            except queue.Empty:
                raise RuntimeError(f"RealSense capture stalled: {self._capture_error or 'frame timeout'}")
        else:
            raw = self.pipeline.wait_for_frames(self.timeout_ms)
        received = time.perf_counter()
        color_raw, depth_raw = raw.get_color_frame(), raw.get_depth_frame()
        if not color_raw or not depth_raw:
            return None
        try:
            color_stamp, color_meta = self.frame_clock(color_raw)
            depth_stamp, depth_meta = self.frame_clock(depth_raw)
        except ValueError:
            if time.monotonic() - self._started < 5:
                return None
            raise
        aligned = self.align.process(raw)
        aligned_at = time.perf_counter()
        color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
        if not color or not depth:
            return None
        result = {
            "color": np.asanyarray(color.get_data()).copy(),
            "depth": np.asanyarray(depth.get_data()).copy(),
            "K": self.K, "D": self.D, "distortion_model": self.distortion_model,
            "depth_scale": self.depth_scale, "color_stamp_ns": color_stamp,
            "depth_stamp_ns": depth_stamp, "color_metadata": color_meta,
            "depth_metadata": depth_meta, "serial": self.serial,
            "timestamp": color_stamp / 1e6,
            "capture_timings_s": {"wait": received-started, "align_and_metadata": aligned_at-received},
        }
        if self._motion_rotations:
            with self._motion_lock:
                result["imu"] = aligned_imu_samples(self._motion_history["accel"],
                                                     self._motion_history["gyro"], color_stamp)
        return result

    def depth_to_xyz(self, depth):
        if np.any(self.D):
            raise ValueError("depth_to_xyz requires rectified pixels when distortion is nonzero")
        v, u = np.indices(depth.shape)
        z = depth.astype(np.float32) * self.depth_scale
        return np.stack(((u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z), axis=-1)

    @staticmethod
    def colorize_depth(depth):
        return cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)

    def stop(self):
        if not self._stopped:
            self._stopped = True
            event = getattr(self, "_capture_stop", None)
            if event is not None:
                event.set()
            try:
                self.pipeline.stop()
            finally:
                thread = getattr(self, "_capture_thread", None)
                if thread is not None:
                    thread.join(timeout=2)


if __name__ == "__main__":
    camera = RealSenseCamera()
    try:
        while True:
            frame = camera.get_frame()
            if frame is not None:
                cv2.imshow("RGB | Depth", cv2.hconcat([frame["color"], camera.colorize_depth(frame["depth"])]))
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
