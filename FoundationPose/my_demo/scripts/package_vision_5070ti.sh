#!/usr/bin/env bash
# Create the transferable source/model bundle from this exact working tree.
set -euo pipefail

_sb_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
_sb_archive="${1:-$_sb_root/runs/vision_5070ti_bundle.tar.gz}"
case "$_sb_archive" in /*) ;; *) echo "Archive path must be absolute" >&2; exit 2 ;; esac
mkdir -p "$(dirname -- "$_sb_archive")"

for _sb_file in \
  "$_sb_root/runs/yolov8/best.pt" \
  "$_sb_root/runs/yolov8/drink_seg_5cls.pt" \
  "$_sb_root/runs/mesh/bottle.obj" \
  "$_sb_root/runs/mesh/nfsq/nfsq.obj" \
  "$_sb_root/runs/mesh/dfsy/dfsy.obj" \
  "$_sb_root/weights/2023-10-28-18-33-37/model_best.pth" \
  "$_sb_root/weights/2024-01-11-20-02-45/model_best.pth"; do
  [ -f "$_sb_file" ] || { echo "Missing asset: $_sb_file" >&2; exit 1; }
done

tar -C "$_sb_root" -czf "$_sb_archive" \
  --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='mycpp/build' --exclude='mycpp/build.*' \
  --exclude='learning/training/runs' \
  my_demo Utils.py estimater.py datareader.py learning mycpp \
  sb_vision_handoff_20260910 \
  runs/mesh/bottle.obj runs/yolov8/best.pt \
  runs/mesh/nfsq/nfsq.obj runs/mesh/dfsy/dfsy.obj runs/yolov8/drink_seg_5cls.pt \
  weights/2023-10-28-18-33-37 \
  weights/2024-01-11-20-02-45
(
  cd "$(dirname -- "$_sb_archive")"
  sha256sum "$(basename -- "$_sb_archive")" > "$(basename -- "$_sb_archive").sha256"
)
sha256sum "$_sb_archive"
printf 'Bundle ready: %s\n' "$_sb_archive"
