#!/usr/bin/env python3
"""Capture raw RGB and aligned depth frames from an Intel D435i for dataset collection.

No YOLO model is loaded and no labels are written. Each run writes one capture
session (one batch); the directory is created on the first saved frame:

    <output-root>/<session>/
    ├── image/frame_000600.png    raw BGR color image
    ├── depth/frame_000600.png    uint16 depth aligned to color, raw sensor units
    ├── K.txt                     3x3 color intrinsics (depth is aligned to color)
    ├── session.json              camera, stream and depth-scale metadata
    └── frames.jsonl              one line per saved frame with timestamps

Keys are read by the OpenCV preview window, so click it before pressing S/A/Q.
Once a labeling tool has written ``labels/`` (with classes.txt), the session
directory can be passed to ``split_dataset.py --source-root``.
Frame numbers start at 600, above the original frame_000001-000540
collection, and continue across all sessions under the same output root, so
sessions can be merged with each other and with the old data without stem clashes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "缺少运行依赖。请激活装有 pyrealsense2、numpy、opencv-python 的环境。\n"
        f"原始错误：{exc}"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "datasets/captures"
DEFAULT_START_INDEX = 600
FRAME_STEM = re.compile(r"^frame_(\d+)$")
SYSTEM_FONT_DIRS = (Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/truetype/liberation"))


def use_system_qt_fonts() -> None:
    """Point Qt at system fonts when the opencv-python wheel's bundled font directory is missing.

    ``import cv2`` sets QT_QPA_FONTDIR to ``cv2/qt/fonts``, which the wheel does not ship, so
    Qt prints "QFontDatabase: Cannot find font directory" on the first ``imshow``.
    """
    current = os.environ.get("QT_QPA_FONTDIR")
    if not current or Path(current).is_dir():
        return
    for candidate in SYSTEM_FONT_DIRS:
        if candidate.is_dir():
            os.environ["QT_QPA_FONTDIR"] = str(candidate)
            return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT, help="所有采集批次的根目录")
    parser.add_argument("--session", help="批次名，例如 shelf_a_light；默认使用当前时间。同名批次会继续追加")
    parser.add_argument("--serial", help="连接多台 RealSense 时指定序列号")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30, help="启动后丢弃的预热帧数，等待自动曝光稳定")
    parser.add_argument("--interval", type=float, default=0.5, help="自动连拍间隔，单位秒")
    parser.add_argument("--auto", action="store_true", help="启动后立即进入自动连拍")
    parser.add_argument("--max-frames", type=int, default=0, help="本次最多保存的帧数，0 表示不限制")
    parser.add_argument(
        "--start-index",
        type=int,
        help=f"手动指定起始帧号；默认从 {DEFAULT_START_INDEX} 开始，已有批次时接着最大帧号继续编",
    )
    parser.add_argument("--max-depth", type=float, default=2.0, help="深度伪彩图的显示上限，单位米，只影响预览")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("--width、--height 和 --fps 必须大于 0")
    if args.warmup < 0 or args.max_frames < 0:
        parser.error("--warmup 和 --max-frames 不能小于 0")
    if args.interval <= 0 or args.max_depth <= 0:
        parser.error("--interval 和 --max-depth 必须大于 0")
    if args.start_index is not None and args.start_index < 1:
        parser.error("--start-index 必须从 1 开始")
    if args.session is not None and (Path(args.session).name != args.session or args.session in {"", ".", ".."}):
        parser.error("--session 只能是单级目录名，不能包含路径分隔符")


def next_frame_index(output_root: Path, first_index: int = DEFAULT_START_INDEX) -> int:
    """Return one past the highest frame number under output_root, but never below first_index."""
    highest = 0
    if output_root.is_dir():
        for path in output_root.glob("*/image/frame_*.png"):
            match = FRAME_STEM.match(path.stem)
            if match:
                highest = max(highest, int(match.group(1)))
    return max(highest + 1, first_index)


def camera_matrix(intrinsics) -> np.ndarray:
    return np.array(
        [[intrinsics.fx, 0.0, intrinsics.ppx], [0.0, intrinsics.fy, intrinsics.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def prepare_session(session_dir: Path, K: np.ndarray, metadata: dict) -> None:
    """Create the session layout, or verify that an existing session used the same camera intrinsics."""
    for name in ("image", "depth"):
        (session_dir / name).mkdir(parents=True, exist_ok=True)

    K_path = session_dir / "K.txt"
    if K_path.exists():
        previous = np.loadtxt(K_path)
        if previous.shape != (3, 3) or not np.allclose(previous, K, atol=1e-3):
            raise ValueError(
                f"批次 {session_dir.name} 已存在，但相机内参不同（分辨率或相机变了）。"
                "请换一个 --session，避免同一批次混用两套内参。"
            )
    else:
        np.savetxt(K_path, K, fmt="%.9f")

    session_json = session_dir / "session.json"
    if not session_json.exists():
        session_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_frame(
    session_dir: Path,
    index: int,
    color_bgr: np.ndarray,
    depth_raw: np.ndarray,
    frame_info: dict,
) -> str:
    """Write one RGB/depth pair and append its record to frames.jsonl. Never overwrites."""
    stem = f"frame_{index:06d}"
    image_path = session_dir / "image" / f"{stem}.png"
    depth_path = session_dir / "depth" / f"{stem}.png"
    if image_path.exists() or depth_path.exists():
        raise FileExistsError(f"拒绝覆盖已有帧：{image_path}（检查 --start-index）")
    if not cv2.imwrite(str(image_path), color_bgr):
        raise OSError(f"RGB 图像保存失败：{image_path}")
    if not cv2.imwrite(str(depth_path), depth_raw):
        image_path.unlink(missing_ok=True)
        raise OSError(f"深度图保存失败：{depth_path}")

    record = {
        "stem": stem,
        **frame_info,
        "depth_valid_ratio": round(float(np.count_nonzero(depth_raw)) / depth_raw.size, 4),
    }
    with (session_dir / "frames.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return stem


def make_depth_visualization(depth_raw: np.ndarray, depth_scale: float, max_depth_m: float) -> np.ndarray:
    depth_m = depth_raw.astype(np.float32) * depth_scale
    normalized = np.clip(depth_m / max_depth_m, 0.0, 1.0)
    visualization = cv2.applyColorMap(np.asarray(normalized * 255.0, dtype=np.uint8), cv2.COLORMAP_JET)
    visualization[depth_raw == 0] = 0
    return visualization


def put_text(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, origin, font, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, font, 0.6, color, 2, cv2.LINE_AA)


def run(args: argparse.Namespace) -> int:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(f"缺少 pyrealsense2，请切换到已安装 RealSense SDK 的环境。原始错误：{exc}") from exc
    use_system_qt_fonts()

    output_root = args.output_root.expanduser().resolve()
    session = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = output_root / session
    next_index = args.start_index or next_frame_index(output_root)

    context = rs.context()
    if len(context.query_devices()) == 0:
        raise RuntimeError("未检测到 RealSense 相机，请检查 USB 3.0、线缆和设备权限。")

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    started = False
    saved = 0
    try:
        profile = pipeline.start(config)
        started = True
        device = profile.get_device()
        depth_scale = device.first_depth_sensor().get_depth_scale()
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        K = camera_matrix(intrinsics)
        metadata = {
            "schema_version": 1,
            "created": datetime.now().isoformat(timespec="seconds"),
            "camera": {
                "name": device.get_info(rs.camera_info.name),
                "serial": device.get_info(rs.camera_info.serial_number),
                "firmware": device.get_info(rs.camera_info.firmware_version),
            },
            "stream": {
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "color_format": "bgr8",
                "depth_format": "z16",
                "depth_aligned_to": "color",
            },
            "depth_scale_m_per_unit": depth_scale,
            "intrinsics": {
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "ppx": intrinsics.ppx,
                "ppy": intrinsics.ppy,
                "model": str(intrinsics.model),
                "coeffs": list(intrinsics.coeffs),
            },
        }
        # A new session directory is only created on the first save, so quitting without
        # saving leaves nothing behind. An existing session is checked up front instead.
        session_ready = session_dir.exists()
        if session_ready:
            prepare_session(session_dir, K, metadata)

        align_to_color = rs.align(rs.stream.color)
        for _ in range(args.warmup):
            pipeline.wait_for_frames()

        print(f"相机：{metadata['camera']['name']}（序列号 {metadata['camera']['serial']}）")
        print(f"批次目录：{session_dir}")
        print(f"起始帧号：frame_{next_index:06d}，深度比例：{depth_scale:.9f} 米/单位")
        print("先用鼠标点一下预览窗口，再按 S/空格保存一帧，按 A 开关自动连拍，按 Q/Esc 退出。")
        print("注意：按键必须按在预览窗口上，在终端里按无效。")

        auto = args.auto
        last_auto_save = 0.0
        flash_text = ""
        flash_until = 0.0
        while True:
            frames = align_to_color.process(pipeline.wait_for_frames())
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            valid_ratio = float(np.count_nonzero(depth_image)) / depth_image.size

            preview = np.hstack(
                (color_image.copy(), make_depth_visualization(depth_image, depth_scale, args.max_depth))
            )
            mode_text = f"AUTO every {args.interval:.1f}s" if auto else "MANUAL"
            put_text(
                preview,
                f"{session} | saved {saved} | next frame_{next_index:06d} | depth valid {valid_ratio:.0%}",
                (12, 28),
                (255, 255, 255),
            )
            put_text(
                preview,
                f"{mode_text} | S/Space: save  A: auto  Q/Esc: quit",
                (12, 56),
                (0, 200, 255) if auto else (255, 255, 255),
            )
            if time.monotonic() < flash_until:
                put_text(preview, flash_text, (12, 84), (0, 255, 0))
            cv2.imshow("D435i Capture", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("a"), ord("A")):
                auto = not auto
                last_auto_save = 0.0
                print("自动连拍：开启" if auto else "自动连拍：关闭")

            now = time.monotonic()
            mode = None
            if key in (ord("s"), ord("S"), ord(" ")):
                mode = "manual"
            elif auto and now - last_auto_save >= args.interval:
                mode = "auto"
                last_auto_save = now
            if mode is None:
                continue

            if not session_ready:
                prepare_session(session_dir, K, metadata)
                session_ready = True
            stem = save_frame(
                session_dir,
                next_index,
                color_image,
                depth_image,
                {
                    "wall_time": datetime.now().isoformat(timespec="milliseconds"),
                    "mode": mode,
                    "color_frame_number": color_frame.get_frame_number(),
                    "color_timestamp_ms": color_frame.get_timestamp(),
                    "depth_timestamp_ms": depth_frame.get_timestamp(),
                },
            )
            saved += 1
            next_index += 1
            flash_text = f"saved {stem}"
            flash_until = now + 0.8
            if mode == "manual":
                print(f"已保存：{stem}")
            if args.max_frames and saved >= args.max_frames:
                print(f"已达到 --max-frames={args.max_frames}，停止采集。")
                break
        return 0
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()
        if saved:
            print(f"本次共保存 {saved} 帧：{session_dir}")
            print(
                "标注到 labels/（含 classes.txt）后可划分：python examples/split_dataset.py "
                f"--source-root {session_dir} --output-root <新的数据集目录>"
            )
        else:
            print("本次没有保存任何帧。")


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
