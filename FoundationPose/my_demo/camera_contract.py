"""Clock and calibration checks shared by the camera and vision processes."""
from __future__ import annotations

import math
import numpy as np


def capture_stamp_ns(timestamp_ms, domain, metadata=None):
    """Use the SDK epoch clock and exposure midpoint metadata when available.

    See librealsense rs_frame.h and D400 metadata-parser.h. Hardware-clock
    timestamps cannot be used as ROS wall time and are rejected.
    """
    domain = str(domain).split(".")[-1]
    if domain not in {"global_time", "system_time"}:
        raise ValueError(f"camera clock is not synchronized: {domain}")
    value = float(timestamp_ms)
    if not math.isfinite(value) or value < 1e12:
        raise ValueError("camera timestamp is not Unix epoch milliseconds")
    metadata = metadata or {}
    offset_us = 0
    basis = "sdk_frame_timestamp"
    if "frame_timestamp" in metadata and "sensor_timestamp" in metadata:
        offset_us = int(metadata["frame_timestamp"]) - int(metadata["sensor_timestamp"])
        if not 0 <= offset_us <= 100000:
            raise ValueError("invalid exposure midpoint offset")
        basis = "exposure_midpoint"
    return int(round(value * 1_000_000)) - offset_us * 1000, basis


def ros_distortion(model, coefficients):
    """Map supported SDK models without inventing zero coefficients."""
    name = str(model).split(".")[-1]
    d = np.asarray(coefficients, dtype=float)
    if d.shape != (5,) or not np.all(np.isfinite(d)):
        raise ValueError("expected five finite SDK distortion coefficients")
    if name == "none":
        if np.any(d != 0):
            raise ValueError("SDK reports distortion none with nonzero coefficients")
        return "plumb_bob", d.tolist()
    if name == "brown_conrady":
        return "plumb_bob", d.tolist()
    if name in {"inverse_brown_conrady", "modified_brown_conrady"} and not np.any(d):
        return "plumb_bob", d.tolist()
    raise ValueError(f"unsupported SDK distortion {name}: {d.tolist()}; rectify explicitly")


def depth_mm_image(raw, scale_m):
    if not math.isfinite(float(scale_m)) or scale_m <= 0:
        raise ValueError("invalid sensor depth scale")
    mm = np.asarray(raw, dtype=np.float64) * (float(scale_m) * 1000.0)
    valid = np.isfinite(mm) & (mm > 0) & (mm <= 65535)
    return np.where(valid, np.rint(mm), 0).astype(np.uint16)


def aligned_imu_samples(accelerations, gyroscopes, color_stamp_ns, *, maximum_image_skew_ns=30_000_000):
    """Select real sampled vectors near exposure time; never retimestamp them."""
    if not accelerations or not gyroscopes:
        return None
    for accel_ns in sorted(accelerations, key=lambda ns: (abs(ns-color_stamp_ns), ns)):
        if abs(accel_ns-color_stamp_ns) > maximum_image_skew_ns:
            break
        gyro_ns = min(gyroscopes, key=lambda ns: (abs(ns-accel_ns), ns))
        if abs(gyro_ns-accel_ns) <= 20_000_000:
            return {"accel": accelerations[accel_ns], "gyro": gyroscopes[gyro_ns],
                    "accel_stamp_ns": accel_ns, "gyro_stamp_ns": gyro_ns}
    return None
