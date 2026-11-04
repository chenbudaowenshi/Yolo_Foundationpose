"""JPEG color and lossless 16UC1 PNG depth, compatible with ROS image_transport.

Depth wire layout follows compressed_depth_image_transport/ConfigHeader:
int32 INV_DEPTH=0, float32 depthParam[2], then the PNG stream. For 16UC1
those inverse-depth parameters are unused. The deployed Linux targets are
little-endian. No range clipping or inverse-depth quantization is performed.
"""
from __future__ import annotations

import struct
import cv2
import numpy as np

COLOR_FORMAT = "rgb8; jpeg compressed bgr8"
DEPTH_FORMAT = "16UC1; compressedDepth png"
DEPTH_HEADER = struct.Struct("<iff")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def bounded_integer(value, minimum, maximum, name):
    if isinstance(value, bool) or not minimum <= int(value) <= maximum or int(value) != float(value):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return int(value)


def encode_color_jpeg(color_bgr, quality=90):
    quality = bounded_integer(quality, 1, 100, "JPEG quality")
    color = np.asarray(color_bgr)
    if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3 or not color.size:
        raise ValueError("JPEG input must be a nonempty uint8 BGR image")
    ok, data = cv2.imencode(".jpg", np.ascontiguousarray(color), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return COLOR_FORMAT, data.tobytes()


def encode_depth_png(depth_mm, compression=3):
    compression = bounded_integer(compression, 0, 9, "PNG compression")
    depth = np.asarray(depth_mm)
    if depth.dtype.kind != "u" or depth.dtype.itemsize != 2 or depth.ndim != 2 or not depth.size:
        raise ValueError("compressedDepth input must be a nonempty uint16 millimetre image")
    ok, data = cv2.imencode(".png", np.ascontiguousarray(depth, dtype=np.uint16),
                          [cv2.IMWRITE_PNG_COMPRESSION, compression])
    if not ok:
        raise ValueError("depth PNG encoding failed")
    return DEPTH_FORMAT, DEPTH_HEADER.pack(0, 0.0, 0.0) + data.tobytes()


def _payload_array(data):
    # ROS uint8[] uses array('B'); memoryview also handles bytes and test fixtures.
    try:
        return np.frombuffer(data, dtype=np.uint8)
    except (TypeError, ValueError) as exc:
        raise ValueError("compressed payload must be a byte buffer") from exc


def decode_color_jpeg(message):
    format_text = " ".join(message.format.strip().lower().split())
    if format_text not in {"jpeg", "rgb8; jpeg compressed bgr8", "bgr8; jpeg compressed bgr8"}:
        raise ValueError(f"unsupported compressed color format: {message.format!r}")
    data = _payload_array(message.data)
    if len(data) < 4 or data[:2].tobytes() != b"\xff\xd8":
        raise ValueError("invalid JPEG payload")
    try:
        color = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError("JPEG decoding failed") from exc
    if color is None or color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("JPEG decoding did not return a BGR image")
    return color  # OpenCV returns BGR; this is the inference input convention.


def decode_depth_png(message):
    format_text = " ".join(message.format.strip().lower().split())
    if format_text not in {"16uc1; compresseddepth", "16uc1; compresseddepth png"}:
        raise ValueError(f"expected lossless 16UC1 compressedDepth PNG, got {message.format!r}")
    data = _payload_array(message.data)
    if len(data) < DEPTH_HEADER.size + len(PNG_SIGNATURE):
        raise ValueError("truncated compressedDepth header")
    compression_format, _, _ = DEPTH_HEADER.unpack_from(data)
    if compression_format != 0:
        raise ValueError("unsupported compressedDepth ConfigHeader format")
    png = data[DEPTH_HEADER.size:]
    if png[:8].tobytes() != PNG_SIGNATURE:
        raise ValueError("compressedDepth payload is not PNG")
    try:
        depth = cv2.imdecode(png, cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        raise ValueError("depth PNG decoding failed") from exc
    if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError("depth PNG must decode to uint16 millimetres")
    return depth


def image_topics(prefix, transport):
    """Subscriber image topics; CameraInfo topic names are transport-independent."""
    prefix = "/" + prefix.strip("/")
    if transport == "raw":
        return prefix + "/color/image_raw", prefix + "/depth/image_rect_raw"
    if transport == "compressed":
        return prefix + "/color/image_raw/compressed", prefix + "/depth/image_rect_raw/compressedDepth"
    raise ValueError("subscriber transport must be raw or compressed")
