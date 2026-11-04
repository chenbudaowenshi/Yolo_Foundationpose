"""ROS/GPU-independent contracts for the robot-facing pose stream."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import numpy as np


@dataclass(frozen=True)
class RGBDFrame:
    stamp_ns: int
    depth_stamp_ns: int
    frame_id: str
    color_bgr: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray

    def validate(self):
        h, w = self.depth_m.shape
        if self.color_bgr.shape != (h, w, 3):
            raise ValueError("RGB/depth dimensions disagree")
        if self.stamp_ns <= 0 or self.depth_stamp_ns <= 0 or not self.frame_id:
            raise ValueError("missing capture time or optical frame")
        if not np.all(np.isfinite(self.K)) or self.K.shape != (3, 3):
            raise ValueError("invalid intrinsics")
        if self.K[0, 0] <= 0 or self.K[1, 1] <= 0:
            raise ValueError("invalid focal length")


def stamp_ns(header):
    return int(header.stamp.sec) * 1000000000 + int(header.stamp.nanosec)


def resize_pose_inputs(frame, max_side, mask=None):
    """Bound registration memory while preserving optical geometry and timestamps."""
    h, w = frame.depth_m.shape
    if max(h, w) <= max_side:
        return frame, mask
    import cv2
    scale = max_side / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    sx, sy = nw / w, nh / h
    K = frame.K.copy()
    K[0, :] *= sx
    K[1, :] *= sy
    K[0, 2] += (sx - 1) / 2
    K[1, 2] += (sy - 1) / 2
    nearest = getattr(cv2, "INTER_NEAREST_EXACT", cv2.INTER_NEAREST)
    reduced = RGBDFrame(frame.stamp_ns, frame.depth_stamp_ns, frame.frame_id,
        cv2.resize(frame.color_bgr, (nw, nh), interpolation=cv2.INTER_AREA),
        cv2.resize(frame.depth_m, (nw, nh), interpolation=nearest), K)
    if mask is not None:
        mask = cv2.resize(np.asarray(mask, np.uint8), (nw, nh), interpolation=nearest).astype(bool)
    return reduced, mask


def decode_image(msg, *, stream=None):
    """Decode ROS Image without cv_bridge, including row padding and endianness."""
    if hasattr(msg, "format"):
        from rgbd_transport import decode_color_jpeg, decode_depth_png
        if stream == "rgb":
            return decode_color_jpeg(msg)
        if stream == "depth":
            return decode_depth_png(msg)
        raise ValueError("compressed image decoding requires rgb/depth stream role")
    formats = {"rgb8": ("u1", 3), "bgr8": ("u1", 3), "16UC1": ("u2", 1), "32FC1": ("f4", 1)}
    if msg.encoding not in formats:
        raise ValueError(f"unsupported image encoding {msg.encoding}")
    kind, channels = formats[msg.encoding]
    dtype = np.dtype((">" if msg.is_bigendian else "<") + kind)
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    if min(h, w) <= 0 or step < w * channels * dtype.itemsize or len(msg.data) < h * step:
        raise ValueError("invalid image dimensions, stride or data length")
    shape = (h, w, channels) if channels > 1 else (h, w)
    strides = (step, channels * dtype.itemsize, dtype.itemsize) if channels > 1 else (step, dtype.itemsize)
    out = np.ndarray(shape, dtype=dtype, buffer=msg.data, strides=strides).astype(dtype.newbyteorder("="), copy=True)
    return out[:, :, ::-1].copy() if msg.encoding == "rgb8" else out


class RGBDSynchronizer:
    """Bounded, one-use pairs with capture-matched RGB and aligned-depth CameraInfo."""
    def __init__(self, frame_id="head_d435i_optical_frame", max_skew_ms=20, capacity=6):
        self.frame_id = frame_id
        self.max_skew_ns = int(max_skew_ms * 1e6)
        self.capacity = capacity
        self.buffers = {name: {} for name in ("rgb", "depth", "rgb_info", "depth_info")}
        self.last_rgb_ns = self.last_depth_ns = 0

    def add(self, channel, msg):
        ns = stamp_ns(msg.header)
        if ns <= 0 or msg.header.frame_id != self.frame_id:
            raise ValueError(f"{channel}: missing stamp or unexpected optical frame")
        if ns <= (self.last_rgb_ns if channel in {"rgb", "rgb_info"} else self.last_depth_ns):
            return
        cache = self.buffers[channel]
        cache[ns] = msg
        while len(cache) > self.capacity:
            del cache[min(cache)]

    def pop_latest(self):
        b = self.buffers
        for cs in sorted(b["rgb"], reverse=True):
            if cs not in b["rgb_info"]:
                continue
            candidates = [ds for ds in b["depth"] if ds in b["depth_info"] and abs(ds - cs) <= self.max_skew_ns]
            if not candidates:
                continue
            ds = min(candidates, key=lambda v: abs(v - cs))
            color, depth = b["rgb"][cs], b["depth"][ds]
            ci, di = b["rgb_info"][cs], b["depth_info"][ds]
            # Consume even malformed pairs, rather than retrying forever.
            for name, cache in b.items():
                cutoff = cs if name in {"rgb", "rgb_info"} else ds
                for ns in list(cache):
                    if ns <= cutoff:
                        del cache[ns]
            self.last_rgb_ns, self.last_depth_ns = cs, ds
            if (ci.width, ci.height) != (di.width, di.height) or not np.allclose(ci.k, di.k, rtol=0, atol=1e-6):
                raise ValueError("depth is not aligned to the RGB calibration")
            if ci.distortion_model != di.distortion_model or len(ci.d) != len(di.d) or not np.allclose(ci.d, di.d):
                raise ValueError("RGB/depth distortion models disagree")
            if ci.distortion_model not in {"plumb_bob", "rational_polynomial"}:
                raise ValueError("unsupported distortion model; rectify before inference")
            K = np.asarray(ci.k, dtype=float).reshape(3, 3)
            D = np.asarray(ci.d, dtype=float)
            if not np.all(np.isfinite(D)):
                raise ValueError("non-finite distortion coefficients")
            color_encoding = "bgr8" if hasattr(color, "format") else color.encoding
            depth_encoding = "16UC1" if hasattr(depth, "format") else depth.encoding
            if color_encoding not in {"rgb8", "bgr8"} or depth_encoding not in {"16UC1", "32FC1"}:
                raise ValueError("incorrect RGB/depth encodings")
            rgb = decode_image(color, stream="rgb")
            depth_m = decode_image(depth, stream="depth").astype(np.float32)
            if rgb.shape[:2] != (ci.height, ci.width) or depth_m.shape != (di.height, di.width):
                raise ValueError("CameraInfo size does not match its decoded image")
            if depth_encoding == "16UC1":
                depth_m *= .001
            depth_m[~np.isfinite(depth_m) | (depth_m <= 0)] = 0
            if np.any(D):
                import cv2
                maps = cv2.initUndistortRectifyMap(K, D, None, K, (ci.width, ci.height), cv2.CV_32FC1)
                rgb = cv2.remap(rgb, *maps, interpolation=cv2.INTER_LINEAR)
                depth_m = cv2.remap(depth_m, *maps, interpolation=cv2.INTER_NEAREST)
            frame = RGBDFrame(cs, ds, self.frame_id, rgb, depth_m, K)
            frame.validate()
            return frame
        return None


def rigid_transform(value):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("pose must be a finite 4x4 matrix")
    R = pose[:3, :3]
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6) or not np.allclose(R.T @ R, np.eye(3), atol=2e-3) or abs(np.linalg.det(R) - 1) > 2e-3:
        raise ValueError("pose is not a rigid transform")
    return pose


def quaternion_xyzw(R):
    # Symmetric eigenproblem also handles rotations close to 180 degrees.
    R = np.asarray(R, float)
    K = np.array([
        [R[0,0]-R[1,1]-R[2,2], R[0,1]+R[1,0], R[0,2]+R[2,0], R[2,1]-R[1,2]],
        [R[0,1]+R[1,0], R[1,1]-R[0,0]-R[2,2], R[1,2]+R[2,1], R[0,2]-R[2,0]],
        [R[0,2]+R[2,0], R[1,2]+R[2,1], R[2,2]-R[0,0]-R[1,1], R[1,0]-R[0,1]],
        [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1], np.trace(R)]]) / 3
    _, vectors = np.linalg.eigh(K)
    q = vectors[:, -1]
    return -q if q[3] < 0 else q


def load_object_profile(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    mesh = (path.parent / config["mesh"]).resolve()
    if hashlib.sha256(mesh.read_bytes()).hexdigest() != config["mesh_sha256"]:
        raise ValueError("mesh checksum differs from object profile; verify dimensions and center")
    if config["mesh_unit"] != "m":
        raise ValueError("object profile must use metres")
    rigid_transform(config["T_model_object_center"])
    config["mesh_path"] = str(mesh)
    return config


def profile_target_class(profile, requested=None):
    """物体配置写了 class_id（nfsq、dfsy）时以它为准；没写时（bottle）默认类别 0。"""
    expected = profile.get("class_id")
    if requested is None:
        return 0 if expected is None else int(expected)
    if expected is not None and int(requested) != int(expected):
        raise ValueError(f"--target-class {requested} 与物体配置 {profile['object_id']} 的类别 {expected} 不一致")
    return int(requested)


def box_iou(a, b):
    lo, hi = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    area = np.prod(np.maximum(hi - lo, 0))
    return float(area / max(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - area, 1e-9))


def assess_quality(frame, detection, pose, bounds, *, min_depth_ratio=.30,
                   min_iou=.35, min_support=.60, tolerance_m=.015):
    """Heuristic pose quality: projection + measured depth in the posed mesh bounds.

    This is not a calibrated probability or covariance estimate. No pose is
    overwritten and no gravity prior is applied.
    """
    pose = rigid_transform(pose)
    mask = np.asarray(detection.mask, bool)
    if mask.shape != frame.depth_m.shape or not np.any(mask):
        return False, 0.0, {"reason": "empty_or_invalid_mask"}
    valid = mask & (frame.depth_m > .05) & np.isfinite(frame.depth_m)
    ratio = float(valid.sum() / mask.sum())
    corners = np.array([[x, y, z] for x in bounds[:, 0] for y in bounds[:, 1] for z in bounds[:, 2]])
    cam = corners @ pose[:3, :3].T + pose[:3, 3]
    if np.any(cam[:, 2] <= .01):
        return False, 0.0, {"reason": "model_behind_camera"}
    pixels = cam @ frame.K.T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    iou = box_iou(np.r_[pixels.min(axis=0), pixels.max(axis=0)], detection.box)
    v, u = np.nonzero(valid)
    step = max(1, len(u) // 2000)
    u, v = u[::step], v[::step]
    z = frame.depth_m[v, u]
    points = np.c_[u, v, np.ones_like(u)] @ np.linalg.inv(frame.K).T * z[:, None]
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    inside = np.all((local >= bounds[0] - tolerance_m) & (local <= bounds[1] + tolerance_m), axis=1)
    support = float(inside.mean()) if len(inside) else 0.0
    geometry_score = math.sqrt(max(0, iou * support * ratio))
    # A conservative weakest-component score, not a probability product of
    # correlated signals. The independent geometry gates below remain required.
    confidence = float(np.clip(min(detection.confidence, geometry_score), 0, 1))
    ok = ratio >= min_depth_ratio and iou >= min_iou and support >= min_support and confidence >= .2
    return ok, confidence, {"reason": "ok" if ok else "pose_quality", "projection_iou": iou,
                            "depth_valid_ratio": ratio, "depth_support_fraction": support}


class PublicationGate:
    """Keep observation time; production injects a PersistentSequence allocator.

    Omitting the allocator is only for standalone, in-memory contract tests.
    """
    def __init__(self, max_delay_s=.30, sequence_allocator=None):
        self.max_delay_ns = int(max_delay_s * 1e9)
        self.last_stamp_ns = 0
        self.sequence_allocator = sequence_allocator
        self.sequence = sequence_allocator.current if sequence_allocator is not None else 0

    def admit(self, frame, now_ns):
        delay = int(now_ns) - frame.stamp_ns
        if delay < -5000000 or delay > self.max_delay_ns:
            return None
        if frame.stamp_ns <= self.last_stamp_ns:
            return None
        if self.sequence >= 2**32 - 1:
            raise RuntimeError("ObjectPose sequence exhausted; reset both endpoints")
        self.sequence = self.sequence_allocator.next() if self.sequence_allocator is not None else self.sequence + 1
        self.last_stamp_ns = frame.stamp_ns
        return self.sequence
