#!/usr/bin/env python3
"""把东方树叶、农夫山泉的 AI 生成模型换算成真实尺寸，供 FoundationPose 使用。

原始模型在 YOLOv8/meshes/<name>/：单位不明（高约 1.15），Y 轴朝上，150 万面带贴图。
每个瓶子的处理：
1. 旋转成 Z 轴朝上、瓶盖朝 +Z（纯旋转，不镜像，标签文字方向不变）。
2. 按卷尺实测高度等比缩放，单位换成米。
3. 原点移到瓶底中心：X/Y 居中，瓶底 Z=0，与 runs/mesh/bottle.obj 约定相同。
4. 输出 runs/mesh/<name>/textured/<name>.obj：全分辨率带贴图，用于查看、存档。
5. 输出 runs/mesh/<name>/<name>.obj：贴图烘焙成顶点颜色并按体素简化，给 FoundationPose 用。
6. 写 my_demo/config/<name>.json：尺寸、几何中心、长轴、缩放比例、SHA256。

用法（pose 环境，在 FoundationPose 目录下）：
  python my_demo/scripts/prepare_drink_meshes.py
  python my_demo/scripts/prepare_drink_meshes.py --only nfsq --nfsq-height-mm 217.5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT.parent / "meshes"
OUTPUT_DIR = ROOT / "runs/mesh"
CONFIG_DIR = ROOT / "my_demo/config"

# 高度来自卷尺照片：农夫山泉 2026-11-04，东方树叶 2026-10-16。
# 源模型目录是 meshes/<name>/，也兼容 meshes/<中文名>/。
DRINKS = {
    "nfsq": {"class_id": 3, "folder": "农夫山泉", "height_mm": 217.0},
    "dfsy": {"class_id": 4, "folder": "东方树叶", "height_mm": 200.0},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_source_obj(name: str, spec: dict) -> Path:
    folders = [SOURCE_DIR / name, SOURCE_DIR / spec["folder"]]
    folder = next((path for path in folders if path.is_dir()), None)
    if folder is None:
        raise SystemExit(f"错误：找不到源模型目录 {folders[0]} 或 {folders[1]}")
    candidates = sorted(folder.glob("*.obj"))
    if len(candidates) != 1:
        raise SystemExit(f"错误：{folder} 顶层应只有一个 OBJ，实际 {len(candidates)} 个")
    return candidates[0]


def load_textured(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False, force="mesh")
    if not isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals) or mesh.visual.material.image is None:
        raise SystemExit(f"错误：{path} 没有读到贴图，检查 material.mtl 和贴图文件是否在同一目录")
    return mesh


def rotate_to_z_up(mesh: trimesh.Trimesh) -> str:
    """最长轴转到 +Z，再保证较窄的瓶盖一端朝上。返回所做旋转的说明。"""
    axis = int(np.argmax(mesh.extents))
    steps = []
    if axis == 0:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2.0, [0.0, 1.0, 0.0]))
        steps.append("绕 Y 轴 -90°（X→Z）")
    elif axis == 1:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2.0, [1.0, 0.0, 0.0]))
        steps.append("绕 X 轴 +90°（Y→Z）")
    z = mesh.vertices[:, 2]
    band = 0.05 * (z.max() - z.min())
    top = np.ptp(mesh.vertices[z > z.max() - band, :2], axis=0).max()
    bottom = np.ptp(mesh.vertices[z < z.min() + band, :2], axis=0).max()
    if top > bottom:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [1.0, 0.0, 0.0]))
        steps.append("绕 X 轴 180°（瓶盖朝上）")
    return "，".join(steps) or "无需旋转"


def move_origin_to_bottom_center(mesh: trimesh.Trimesh) -> None:
    (x0, y0, z0), (x1, y1, _) = mesh.bounds
    mesh.apply_translation([-(x0 + x1) / 2.0, -(y0 + y1) / 2.0, -z0])


def bake_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    """按每个顶点的 UV 双线性采样贴图，得到 (N, 3) uint8 RGB。"""
    image = np.asarray(mesh.visual.material.image.convert("RGB"))
    height, width = image.shape[:2]
    uv = np.asarray(mesh.visual.uv, dtype=np.float64)
    count = len(uv)
    # cv2.remap 要求行列数都小于 32767，所以把顶点排成每行 1024 个的二维表。
    rows = -(-count // 1024)
    padded = np.zeros((rows * 1024, 2), dtype=np.float64)
    padded[:count] = uv
    map_x = (padded[:, 0] * width - 0.5).astype(np.float32).reshape(rows, 1024)
    map_y = ((1.0 - padded[:, 1]) * height - 0.5).astype(np.float32).reshape(rows, 1024)
    colors = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return colors.reshape(-1, 3)[:count]


def simplify_with_colors(mesh: trimesh.Trimesh, colors: np.ndarray, voxel_m: float) -> trimesh.Trimesh:
    """体素聚类：每个体素内的顶点合并成一个，位置和颜色取平均。"""
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    o3d_mesh = o3d_mesh.simplify_vertex_clustering(
        voxel_size=voxel_m, contraction=o3d.geometry.SimplificationContraction.Average
    )
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_unreferenced_vertices()
    if not o3d_mesh.has_vertex_colors():
        raise RuntimeError("简化后丢失了顶点颜色")
    simplified = trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        vertex_colors=np.round(np.asarray(o3d_mesh.vertex_colors) * 255.0).astype(np.uint8),
        process=False,
    )
    move_origin_to_bottom_center(simplified)
    return simplified


def write_config(name: str, spec: dict, source: Path, raw_extents: np.ndarray, scale: float,
                 rotation: str, fp_mesh_path: Path, textured_path: Path, fp_mesh: trimesh.Trimesh) -> Path:
    extents = [round(float(value), 6) for value in fp_mesh.extents]
    height = extents[2]
    config = {
        "object_id": name,
        "class_id": spec["class_id"],
        "display_name": spec["folder"],
        "mesh": str(Path("../../runs/mesh") / name / fp_mesh_path.name),
        "mesh_textured": str(Path("../../runs/mesh") / name / "textured" / textured_path.name),
        "mesh_unit": "m",
        "mesh_sha256": sha256(fp_mesh_path),
        "expected_extents_m": extents,
        "T_model_object_center": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, round(height / 2.0, 6)],
            [0, 0, 0, 1],
        ],
        "long_axis_in_object_frame": [0, 0, 1],
        "nominal_diameter_m": round(float(np.mean(extents[:2])), 6),
        "height_m": height,
        "mass_kg": None,
        "center_definition": "center of model bounds; origin at bottom center, Z up, cap at +Z",
        "source": {
            "mesh": str(source),
            "mesh_sha256": sha256(source),
            "raw_extents": [round(float(value), 6) for value in raw_extents],
            "rotation": rotation,
        },
        "scale": {
            "method": "uniform, from tape-measured height",
            "measured_height_m": spec["height_mm"] / 1000.0,
            "meters_per_source_unit": round(scale, 9),
        },
    }
    path = CONFIG_DIR / f"{name}.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def process(name: str, spec: dict, voxel_mm: float) -> None:
    source = find_source_obj(name, spec)
    print(f"\n=== {name}（{spec['folder']}）: {source}")
    mesh = load_textured(source)
    raw_extents = mesh.extents.copy()
    print(f"原始尺寸（原单位）: {np.round(raw_extents, 5)}，{len(mesh.vertices)} 顶点 / {len(mesh.faces)} 面")

    rotation = rotate_to_z_up(mesh)
    scale = spec["height_mm"] / 1000.0 / float(mesh.extents[2])
    mesh.apply_scale(scale)
    move_origin_to_bottom_center(mesh)
    print(f"旋转: {rotation}；缩放: 1 原单位 = {scale * 1000.0:.3f} mm（实测高 {spec['height_mm']:.1f} mm）")

    out_dir = OUTPUT_DIR / name
    textured_path = out_dir / "textured" / f"{name}.obj"
    textured_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(textured_path)

    fp_mesh = simplify_with_colors(mesh, bake_vertex_colors(mesh), voxel_mm / 1000.0)
    fp_mesh_path = out_dir / f"{name}.obj"
    fp_mesh.export(fp_mesh_path)

    config_path = write_config(name, spec, source, raw_extents, scale, rotation, fp_mesh_path, textured_path, fp_mesh)
    full = mesh.extents * 1000.0
    small = fp_mesh.extents * 1000.0
    print(f"全分辨率带贴图: {textured_path}  尺寸 {full[0]:.1f} × {full[1]:.1f} × {full[2]:.1f} mm")
    print(f"FoundationPose 用: {fp_mesh_path}  尺寸 {small[0]:.1f} × {small[1]:.1f} × {small[2]:.1f} mm，"
          f"{len(fp_mesh.vertices)} 顶点 / {len(fp_mesh.faces)} 面")
    print(f"配置: {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=sorted(DRINKS), help="只处理一个瓶子")
    for name, spec in DRINKS.items():
        parser.add_argument(f"--{name}-height-mm", type=float, default=spec["height_mm"],
                            help=f"{spec['folder']} 实测总高度（默认 {spec['height_mm']} mm）")
    parser.add_argument("--voxel-mm", type=float, default=1.0,
                        help="FoundationPose 版本的简化体素边长，越小越精细、越慢（默认 1.0 mm）")
    args = parser.parse_args()
    if not 0.2 <= args.voxel_mm <= 5.0:
        parser.error("--voxel-mm 应在 0.2–5 之间")

    for name, spec in DRINKS.items():
        if args.only and name != args.only:
            continue
        height = getattr(args, f"{name}_height_mm")
        if not 50.0 <= height <= 400.0:
            parser.error(f"--{name}-height-mm={height} 不像瓶子高度（应为 50–400 mm）")
        process(name, {**spec, "height_mm": height}, args.voxel_mm)


if __name__ == "__main__":
    main()
