#!/usr/bin/env python3
"""Show real-time YOLO detections and aligned depth from an Intel D435i."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def ensure_fopose_python() -> None:
    """Use the environment already configured for both RealSense and YOLO."""
    if os.environ.get("D435I_FOPOSE_REEXEC") == "1":
        return
    expected = Path(
        os.environ.get("FOPOSE_PYTHON", "/home/hssw/miniconda3/envs/fopose/bin/python")
    ).expanduser()
    if not expected.is_file():
        return
    try:
        current = Path(sys.executable).resolve()
        target = expected.resolve()
    except OSError:
        return
    if current == target:
        return
    environment = os.environ.copy()
    environment["D435I_FOPOSE_REEXEC"] = "1"
    environment.pop("PYTHONHOME", None)
    print(f"Switching Python interpreter: {current} -> {target}", flush=True)
    os.execve(str(target), [str(target), *sys.argv], environment)


ensure_fopose_python()

try:
    import cv2
    import numpy as np
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit(
        "缺少运行依赖。请激活 yolov8 环境并安装 pyrealsense2、numpy、opencv-python。\n"
        f"原始错误：{exc}"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "runs/segment/drink_seg/weights/best.pt"
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "runs/d435i_seg_diagnostic/snapshots"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="可信的 YOLO .pt 权重")
    parser.add_argument("--serial", help="连接多台 RealSense 时指定序列号")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30, help="启动后丢弃的预热帧数")
    parser.add_argument("--device", default="0", help="YOLO 推理设备，例如 0 或 cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--half", action="store_true", help="在 CUDA 上使用 FP16 推理")
    parser.add_argument("--inner-ratio", type=float, default=0.5, help="框内用于测距的中心区域比例")
    parser.add_argument("--min-depth", type=float, default=0.1, help="有效最小深度，单位米")
    parser.add_argument("--max-depth", type=float, default=5.0, help="有效最大深度，单位米")
    parser.add_argument("--no-depth-view", action="store_true", help="不显示右侧深度伪彩图")
    parser.add_argument("--no-mask-view", action="store_true", help="仅显示框，不叠加 YOLO-seg 掩膜")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument(
        "--trust-checkpoints",
        action="store_true",
        help="允许当前进程完整加载可信的旧版 YOLO .pt 权重",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.width <= 0 or args.height <= 0 or args.fps <= 0 or args.imgsz <= 0:
        parser.error("--width、--height、--fps 和 --imgsz 必须大于 0")
    if args.warmup < 0:
        parser.error("--warmup 不能小于 0")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        parser.error("--conf 和 --iou 必须位于 [0, 1]")
    if not 0.0 < args.inner_ratio <= 1.0:
        parser.error("--inner-ratio 必须位于 (0, 1]")
    if args.min_depth < 0 or args.max_depth <= args.min_depth:
        parser.error("深度范围无效，要求 0 <= --min-depth < --max-depth")
    if args.half and str(args.device).lower() == "cpu":
        parser.error("--half 只能用于 CUDA 推理")
    if not args.trust_checkpoints:
        parser.error(
            "当前仓库需要完整反序列化旧版 .pt；确认权重由你自己训练或来源可信后，"
            "请添加 --trust-checkpoints"
        )


def make_depth_visualization(depth_raw: np.ndarray, depth_scale: float, max_depth_m: float) -> np.ndarray:
    depth_m = depth_raw.astype(np.float32) * depth_scale
    normalized = np.clip(depth_m / max_depth_m, 0.0, 1.0)
    gray = np.asarray(normalized * 255.0, dtype=np.uint8)
    visualization = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    visualization[depth_raw == 0] = 0
    return visualization


def estimate_box_depth(
    depth_raw: np.ndarray,
    box_xyxy: np.ndarray,
    depth_scale: float,
    inner_ratio: float,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[float | None, tuple[int, int, int, int]]:
    """Return robust median depth and the inner ROI used for measurement."""
    height, width = depth_raw.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(x1 + 1, min(width, int(x2)))
    y2 = max(y1 + 1, min(height, int(y2)))

    margin_x = int((x2 - x1) * (1.0 - inner_ratio) / 2.0)
    margin_y = int((y2 - y1) * (1.0 - inner_ratio) / 2.0)
    rx1, ry1 = x1 + margin_x, y1 + margin_y
    rx2, ry2 = x2 - margin_x, y2 - margin_y
    roi = depth_raw[ry1:ry2, rx1:rx2].astype(np.float32) * depth_scale
    valid = roi[np.isfinite(roi) & (roi >= min_depth_m) & (roi <= max_depth_m)]
    if valid.size < 20:
        return None, (rx1, ry1, rx2, ry2)

    q1, q3 = np.percentile(valid, [25.0, 75.0])
    iqr = q3 - q1
    if iqr > 0:
        valid = valid[(valid >= q1 - 1.5 * iqr) & (valid <= q3 + 1.5 * iqr)]
    distance = float(np.median(valid)) if valid.size else None
    return distance, (rx1, ry1, rx2, ry2)


def class_color(class_id: int) -> tuple[int, int, int]:
    colors = (
        (0, 255, 0),
        (255, 128, 0),
        (0, 128, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
    )
    return colors[class_id % len(colors)]


def draw_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    y = max(text_height + baseline + 4, y)
    cv2.rectangle(
        image,
        (x, y - text_height - baseline - 4),
        (x + text_width + 6, y + 2),
        color,
        -1,
    )
    cv2.putText(image, text, (x + 3, y - baseline), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def annotate_detections(
    color_image: np.ndarray,
    depth_image: np.ndarray,
    result,
    names: dict[int, str],
    depth_scale: float,
    args: argparse.Namespace,
) -> np.ndarray:
    annotated = color_image.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return annotated

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    masks = None
    if not args.no_mask_view and result.masks is not None:
        masks = result.masks.data.detach().float().cpu().numpy()
    for detection_index, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
        x1, y1, x2, y2 = (int(value) for value in box)
        color = class_color(class_id)
        if masks is not None and detection_index < len(masks):
            mask = np.squeeze(masks[detection_index])
            if mask.shape != depth_image.shape:
                mask = cv2.resize(
                    mask,
                    (depth_image.shape[1], depth_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            foreground = mask > 0.5
            if np.any(foreground):
                color_array = np.asarray(color, dtype=np.float32)
                annotated[foreground] = np.clip(
                    annotated[foreground].astype(np.float32) * 0.55 + color_array * 0.45,
                    0,
                    255,
                ).astype(np.uint8)
                contours, _ = cv2.findContours(
                    foreground.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(annotated, contours, -1, color, 2)
        distance, roi = estimate_box_depth(
            depth_image,
            box,
            depth_scale,
            args.inner_ratio,
            args.min_depth,
            args.max_depth,
        )
        class_name = names[class_id]
        depth_text = "depth=N/A" if distance is None else f"{distance:.2f}m"
        text = f"{class_name} {confidence:.2f} {depth_text}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        rx1, ry1, rx2, ry2 = roi
        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), color, 1)
        draw_label(annotated, text, (max(0, x1), y1 - 4), color)
    return annotated


def save_snapshot(
    snapshot_dir: Path,
    color_raw: np.ndarray,
    annotated: np.ndarray,
    depth_raw: np.ndarray,
) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    color_path = snapshot_dir / f"{timestamp}_rgb.png"
    annotated_path = snapshot_dir / f"{timestamp}_annotated.png"
    depth_path = snapshot_dir / f"{timestamp}_depth.png"
    if not cv2.imwrite(str(color_path), color_raw):
        raise OSError(f"RGB 截图保存失败：{color_path}")
    if not cv2.imwrite(str(annotated_path), annotated):
        color_path.unlink(missing_ok=True)
        raise OSError(f"截图保存失败：{annotated_path}")
    if not cv2.imwrite(str(depth_path), depth_raw):
        color_path.unlink(missing_ok=True)
        annotated_path.unlink(missing_ok=True)
        raise OSError(f"深度图保存失败：{depth_path}")
    print(f"已保存 RGB、标注预览和深度图：{snapshot_dir / timestamp}")


def run(args: argparse.Namespace) -> int:
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"模型不存在：{model_path}")

    # PyTorch 2.7 + Ultralytics 8.0.29 compatibility, scoped to this process.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    # When this file is executed as ``python examples/d435i_realtime.py``,
    # sys.path starts at examples/ rather than the repository root containing
    # the local ultralytics package.
    sys.path.insert(0, str(REPO_ROOT))
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    context = rs.context()
    devices = [device for device in context.query_devices()]
    if not devices:
        raise RuntimeError("未检测到 RealSense 相机，请检查 USB 3.0、线缆和设备权限。")

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    started = False
    try:
        profile = pipeline.start(config)
        started = True
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align_to_color = rs.align(rs.stream.color)
        for _ in range(args.warmup):
            pipeline.wait_for_frames()

        print(f"模型：{model_path}")
        print(f"设备：{args.device}，深度比例：{depth_scale:.9f} 米/单位")
        print("按 S/空格保存标注截图和原始深度图，按 Q/Esc 退出。")

        smoothed_fps = 0.0
        previous_time = time.perf_counter()
        while True:
            frames = align_to_color.process(pipeline.wait_for_frames())
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            inference_start = time.perf_counter()
            result = model.predict(
                color_image,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                half=args.half,
                retina_masks=True,
                verbose=False,
            )[0]
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            annotated = annotate_detections(color_image, depth_image, result, model.names, depth_scale, args)
            object_count = 0 if result.boxes is None else len(result.boxes)

            now = time.perf_counter()
            instantaneous_fps = 1.0 / max(now - previous_time, 1e-9)
            previous_time = now
            smoothed_fps = instantaneous_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instantaneous_fps
            cv2.putText(
                annotated,
                f"FPS {smoothed_fps:.1f} | inference {inference_ms:.1f} ms | objects {object_count}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if args.no_depth_view:
                preview = annotated
            else:
                depth_visualization = make_depth_visualization(depth_image, depth_scale, args.max_depth)
                preview = np.hstack((annotated, depth_visualization))
            cv2.imshow("D435i YOLO RGB-D Realtime", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S"), ord(" ")):
                save_snapshot(
                    args.snapshot_dir.expanduser().resolve(),
                    color_image,
                    annotated,
                    depth_image,
                )
        return 0
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n用户中止。")
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
