#!/usr/bin/env bash
set -euo pipefail

bundle_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${FOPOSE_PYTHON:-python}"
foundationpose_root="${FOUNDATIONPOSE_ROOT:-${bundle_root}/FoundationPose}"

if [[ ! -f "${foundationpose_root}/estimater.py" ]]; then
  echo "ERROR: FoundationPose not found: ${foundationpose_root}" >&2
  echo "Set FOUNDATIONPOSE_ROOT or put FoundationPose inside this bundle." >&2
  exit 1
fi

exec "${python_bin}" "${bundle_root}/examples/d435i_autonomous_vision.py" \
  --foundationpose-root "${foundationpose_root}" \
  "$@"

