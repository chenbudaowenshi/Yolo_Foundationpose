#!/usr/bin/env bash
set -euo pipefail

_sb_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export SB_VISION_PYTHON="${SB_VISION_PYTHON:-${SB_5070TI_ENV:-$_sb_root/.envs/sb_vision_5070ti}/bin/python}"
if [ ! -x "$SB_VISION_PYTHON" ]; then
  echo "Vision Python is missing: $SB_VISION_PYTHON; run install_vision_5070ti.sh first" >&2
  exit 2
fi
exec bash "$_sb_root/my_demo/scripts/run_sb_vision.sh" "$@"
