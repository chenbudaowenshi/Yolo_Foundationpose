#!/usr/bin/env python3
"""Live test of a YOLO segmentation model on an Intel D435i: masks, class, confidence and distance.

Run with the pose environment (Ultralytics 8.4 + pyrealsense2):
    python examples/d435i_seg_live.py
    python examples/d435i_seg_live.py --model runs/segment/drink_seg_5cls/weights/best.pt --conf 0.38

Each object shows its class, confidence (%) and distance; the bottom panel shows the model's test-set accuracy.
The right pane previews the binary masks of --mask-classes (default nfsq, dfsy); S saves one 0/255 png per object
plus rgb, aligned depth and K.txt, the inputs FoundationPose needs.
Keys: Q/Esc quit, S/Space save snapshot + masks, +/- confidence threshold, M side pane mask/depth, D hide side pane.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    import cv2
    import numpy as np
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit(f"缺少运行依赖，请先执行 conda activate pose。原始错误：{exc}") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "runs/segment/drink_seg_5cls/weights/best.pt"
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "runs/segment/live_snapshots"
WINDOW = "D435i YOLO-seg live"
FONT = cv2.FONT_HERSHEY_SIMPLEX
# BGR, close to each drink's own colour so the overlay reads at a glance
CLASS_COLORS = {
    "fenda": (0, 140, 255),
    "cola": (255, 90, 0),
    "qixi": (0, 220, 0),
    "nfsq": (0, 0, 255),
    "dfsy": (0, 220, 255),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLO-seg 权重 .pt")
    parser.add_argument("--serial", help="连接多台 RealSense 时指定序列号")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30, help="启动后丢弃的帧数，等自动曝光稳定")
    parser.add_argument("--device", default="0", help="推理设备，例如 0 或 cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.38, help="置信度阈值（验证集 F1 最高点约 0.38）")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument("--half", action="store_true", help="在 CUDA 上用 FP16 推理")
    parser.add_argument("--min-depth", type=float, default=0.1, help="有效深度下限，单位米")
    parser.add_argument("--max-depth", type=float, default=4.0, help="有效深度上限，单位米")
    parser.add_argument("--mask-classes", nargs="+", default=["nfsq", "dfsy"],
                        help="按 S 截图时为这些类别各导出一张黑白 mask（默认农夫山泉、东方树叶）")
    parser.add_argument("--side-view", choices=("mask", "depth"), default="mask",
                        help="右侧显示 mask 预览还是深度图（运行中按 M 切换）")
    parser.add_argument("--no-side-view", "--no-depth-view", dest="no_side_view", action="store_true",
                        help="启动时不显示右侧视图（运行中按 D 切换）")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    return parser


def load_model(path: Path):
    if not path.is_file():
        raise SystemExit(f"模型不存在：{path}")
    import ultralytics
    from ultralytics import YOLO

    # The repo ships Ultralytics 8.0.29, which cannot read checkpoints trained with 8.4.
    if Path(ultralytics.__file__).resolve().is_relative_to(REPO_ROOT):
        raise SystemExit(f"导入的是仓库自带的旧版 Ultralytics {ultralytics.__version__}，请用 pose 环境运行本脚本。")
    model = YOLO(str(path))
    if model.task != "segment":
        raise SystemExit(f"{path.name} 不是分割模型（task={model.task}）")
    return model


def open_camera(args: argparse.Namespace):
    if len(rs.context().query_devices()) == 0:
        raise SystemExit("未检测到 RealSense 相机，请检查 USB 3.0 线缆和接口。")
    pipeline, config = rs.pipeline(), rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    # depth is aligned to the colour stream, so the colour intrinsics describe rgb, depth and masks alike
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]])
    align = rs.align(rs.stream.color)
    for _ in range(args.warmup):
        pipeline.wait_for_frames()
    return pipeline, align, depth_scale, K


def read_frames(pipeline, align):
    """Return (color BGR, depth aligned to color), or None when a stream is missing."""
    frames = align.process(pipeline.wait_for_frames(5000))
    depth, color = frames.get_depth_frame(), frames.get_color_frame()
    if not depth or not color:
        return None
    return np.asanyarray(color.get_data()).copy(), np.asanyarray(depth.get_data()).copy()


def mask_distance(depth_raw: np.ndarray, mask: np.ndarray, depth_scale: float, min_m: float, max_m: float) -> float | None:
    """Median depth inside the mask, eroded first so edge pixels do not pick up the background."""
    inner = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if inner.sum() >= 30:
        mask = inner
    values = depth_raw[mask].astype(np.float32) * depth_scale
    values = values[(values >= min_m) & (values <= max_m)]
    return float(np.median(values)) if values.size >= 30 else None


def draw_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    (w, h), baseline = cv2.getTextSize(text, FONT, 0.55, 2)
    x, y = origin
    x = max(0, min(x, image.shape[1] - w - 6))  # keep labels of bottles at the right edge inside the frame
    y = max(h + baseline + 4, y)
    cv2.rectangle(image, (x, y - h - baseline - 4), (x + w + 6, y + 2), color, -1)
    cv2.putText(image, text, (x + 3, y - baseline), FONT, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def annotate(color: np.ndarray, depth_raw: np.ndarray, result, depth_scale: float, args: argparse.Namespace):
    """Draw masks, contours and labels; return the image, the detections and their masks (same order)."""
    out = color.copy()
    detections, labels, masks_out = [], [], []
    if result.boxes is None or len(result.boxes) == 0:
        return out, detections, masks_out
    classes = result.boxes.cls.int().tolist()
    confs = result.boxes.conf.tolist()
    boxes = result.boxes.xyxy.int().tolist()
    masks = result.masks.data.cpu().numpy() > 0.5 if result.masks is not None else [None] * len(classes)
    for cls, conf, box, mask in zip(classes, confs, boxes, masks):
        name = result.names[cls]
        bgr = CLASS_COLORS.get(name, (255, 0, 255))
        distance = None
        if mask is not None:
            if mask.shape != depth_raw.shape:
                mask = cv2.resize(mask.astype(np.uint8), depth_raw.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
            out[mask] = (0.55 * out[mask] + 0.45 * np.array(bgr)).astype(np.uint8)
            contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(out, contours, -1, bgr, 2)
            distance = mask_distance(depth_raw, mask, depth_scale, args.min_depth, args.max_depth)
        else:
            cv2.rectangle(out, tuple(box[:2]), tuple(box[2:]), bgr, 2)
        labels.append((f"{name} {conf:.0%} " + ("--" if distance is None else f"{distance:.2f}m"), (box[0], box[1] - 4), bgr))
        detections.append({"class": name, "conf": round(conf, 3), "distance_m": None if distance is None else round(distance, 3),
                           "box_xyxy": box})
        masks_out.append(mask)
    for text, origin, bgr in labels:  # labels last so no mask is painted over them
        draw_label(out, text, origin, bgr)
    return out, detections, masks_out


def mask_view(shape: tuple[int, ...], detections: list[dict], masks: list, mask_classes: list[str]) -> np.ndarray:
    """Preview exactly what S saves: white object, black background, one label per exported mask."""
    view = np.zeros((shape[0], shape[1], 3), np.uint8)
    picked = [(d, m) for d, m in zip(detections, masks) if d["class"] in mask_classes and m is not None]
    for _, mask in picked:
        view[mask] = 255
    for det, _ in picked:
        draw_label(view, f"{det['class']} mask", (det["box_xyxy"][0], det["box_xyxy"][1] - 4),
                   CLASS_COLORS.get(det["class"], (255, 0, 255)))
    if not picked:
        cv2.putText(view, f"no {' / '.join(mask_classes)} in view", (10, 30), FONT, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return view


def load_val_accuracy(model_path: Path) -> str | None:
    """Test-set accuracy of the training run: the results.csv row with the best mask mAP50-95."""
    csv_path = model_path.parent.parent / "results.csv"
    if not csv_path.is_file():
        return None
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows or "metrics/mAP50-95(M)" not in rows[0]:
        return None
    best = max(rows, key=lambda r: float(r["metrics/mAP50-95(M)"]))
    return "test set: precision {:.1%}  recall {:.1%}  mAP50 {:.1%}".format(
        float(best["metrics/precision(M)"]), float(best["metrics/recall(M)"]), float(best["metrics/mAP50(M)"]))


def draw_hud(image: np.ndarray, fps: float, infer_ms: float, conf: float, detections: list[dict],
             val_accuracy: str | None = None) -> None:
    counts = Counter(d["class"] for d in detections)
    lines = [f"FPS {fps:.1f}  infer {infer_ms:.0f} ms  conf>={conf:.0%}"]
    if val_accuracy:
        lines.append(val_accuracy)
    lines.append("  ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "no objects")
    # Darken a panel instead of outlining the text: OpenCV 5 glyph width depends on thickness,
    # so a thick black copy under thin white text no longer lines up.
    width = max(cv2.getTextSize(t, FONT, 0.6, 1)[0][0] for t in lines) + 16
    top = image.shape[0] - 24 * len(lines) - 8
    panel = image[top:, :min(width, image.shape[1])]
    panel[:] = (panel * 0.35).astype(np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(image, text, (8, top + 22 + 24 * i), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def depth_colormap(depth_raw: np.ndarray, depth_scale: float, max_m: float) -> np.ndarray:
    gray = (np.clip(depth_raw * depth_scale / max_m, 0, 1) * 255).astype(np.uint8)
    view = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    view[depth_raw == 0] = 0  # no depth measured
    return view


def save_snapshot(folder: Path, color: np.ndarray, annotated: np.ndarray, depth_raw: np.ndarray,
                  detections: list[dict], masks: list, mask_classes: list[str], depth_scale: float, K: np.ndarray,
                  conf: float) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    cv2.imwrite(str(folder / f"{stamp}_rgb.png"), color)
    cv2.imwrite(str(folder / f"{stamp}_annotated.png"), annotated)
    cv2.imwrite(str(folder / f"{stamp}_depth.png"), depth_raw)  # 16-bit raw depth, multiply by depth_scale for metres
    np.savetxt(folder / f"{stamp}_K.txt", K, fmt="%.9f")
    # one 0/255 png per object, same size as rgb; index 0 is the most confident one of its class
    saved = Counter()
    for det, mask in zip(detections, masks):
        name = det["class"]
        if name in mask_classes and mask is not None:
            path = folder / f"{stamp}_mask_{name}_{saved[name]}.png"
            cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
            det["mask_file"] = path.name
            saved[name] += 1
    (folder / f"{stamp}_detections.json").write_text(json.dumps(
        {"depth_scale": depth_scale, "K": K.tolist(), "conf_threshold": conf, "mask_classes": mask_classes,
         "detections": detections}, ensure_ascii=False, indent=1))
    print(f"已保存截图：{folder / stamp}_*  ({len(detections)} 个目标)")
    if saved:
        print("  mask：" + "  ".join(f"{name}×{n}" for name, n in saved.items()))
    else:
        print(f"  画面里没有 {' / '.join(mask_classes)}，本次没有生成 mask")


def run(args: argparse.Namespace) -> int:
    model_path = args.model.expanduser().resolve()
    model = load_model(model_path)
    unknown = sorted(set(args.mask_classes) - set(model.names.values()))
    if unknown:
        raise SystemExit(f"--mask-classes 里有模型不认识的类别 {unknown}，可选：{list(model.names.values())}")
    val_accuracy = load_val_accuracy(model_path)
    pipeline, align, depth_scale, K = open_camera(args)
    conf, show_side, side = args.conf, not args.no_side_view, args.side_view
    fps, last = 0.0, time.perf_counter()
    print(f"模型：{model_path}  类别：{list(model.names.values())}")
    if val_accuracy:
        print(f"测试集成绩：{val_accuracy}")
    print(f"导出 mask 的类别：{args.mask_classes}，保存到 {args.snapshot_dir.expanduser().resolve()}")
    print("按键：Q/Esc 退出 | S/空格 截图并导出 mask | +/- 调整置信度 | M 右侧切换 mask/深度 | D 显示/隐藏右侧")
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            frames = read_frames(pipeline, align)
            if frames is None:
                continue
            color, depth_raw = frames
            start = time.perf_counter()
            result = model.predict(color, imgsz=args.imgsz, conf=conf, iou=args.iou, device=args.device,
                                   half=args.half, retina_masks=True, verbose=False)[0]
            infer_ms = (time.perf_counter() - start) * 1000
            annotated, detections, masks = annotate(color, depth_raw, result, depth_scale, args)
            now = time.perf_counter()
            fps = 1 / max(now - last, 1e-6) if fps == 0 else 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            draw_hud(annotated, fps, infer_ms, conf, detections, val_accuracy)
            if show_side:
                pane = (mask_view(color.shape, detections, masks, args.mask_classes) if side == "mask"
                        else depth_colormap(depth_raw, depth_scale, args.max_depth))
                view = np.hstack([annotated, pane])
            else:
                view = annotated
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("s"), ord("S"), ord(" ")):
                save_snapshot(args.snapshot_dir.expanduser().resolve(), color, annotated, depth_raw, detections, masks,
                              args.mask_classes, depth_scale, K, conf)
            elif key in (ord("+"), ord("=")):
                conf = min(round(conf + 0.05, 2), 0.95)
            elif key in (ord("-"), ord("_")):
                conf = max(round(conf - 0.05, 2), 0.05)
            elif key in (ord("m"), ord("M")):
                side = "depth" if side == "mask" else "mask"
                show_side = True
            elif key in (ord("d"), ord("D")):
                show_side = not show_side
        return 0
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not 0.0 < args.conf < 1.0:
        parser.error("--conf 必须在 (0, 1) 之间")
    if not 0.0 <= args.min_depth < args.max_depth:
        parser.error("深度范围无效，要求 0 <= --min-depth < --max-depth")
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n用户中止。")
        return 0
    except RuntimeError as exc:  # e.g. camera unplugged: "Frame didn't arrive within 5000"
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
