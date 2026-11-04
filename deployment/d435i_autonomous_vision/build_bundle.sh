#!/usr/bin/env bash
set -euo pipefail

packaging_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${packaging_dir}/../.." && pwd)"
foundationpose_source=""
output_path=""

usage() {
  cat <<'EOF'
Usage: build_bundle.sh [--with-foundationpose PATH] [--output ARCHIVE.tar.gz]

By default, creates an application-only archive in dist/. FoundationPose may
be included only when its non-commercial license is suitable for the delivery.
EOF
}

while (($#)); do
  case "$1" in
    --with-foundationpose)
      [[ $# -ge 2 ]] || { echo "ERROR: missing FoundationPose path" >&2; exit 2; }
      foundationpose_source="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || { echo "ERROR: missing output path" >&2; exit 2; }
      output_path="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

variant="app"
if [[ -n "${foundationpose_source}" ]]; then
  variant="full"
  foundationpose_source="$(cd -- "${foundationpose_source}" && pwd)"
  [[ -f "${foundationpose_source}/estimater.py" ]] || {
    echo "ERROR: estimater.py not found under ${foundationpose_source}" >&2
    exit 1
  }
  [[ -f "${foundationpose_source}/LICENSE" ]] || {
    echo "ERROR: FoundationPose LICENSE is missing" >&2
    exit 1
  }
fi

if [[ -z "${output_path}" ]]; then
  mkdir -p "${repo_root}/dist"
  output_path="${repo_root}/dist/d435i-autonomous-vision-${variant}.tar.gz"
else
  output_parent="$(dirname -- "${output_path}")"
  mkdir -p "${output_parent}"
  output_path="$(cd -- "${output_parent}" && pwd)/$(basename -- "${output_path}")"
fi

[[ "${output_path}" == *.tar.gz ]] || {
  echo "ERROR: output must end with .tar.gz" >&2
  exit 2
}
[[ ! -e "${output_path}" && ! -e "${output_path}.sha256" ]] || {
  echo "ERROR: refusing to overwrite existing output: ${output_path}" >&2
  exit 1
}

stage_root="$(mktemp -d)"
trap 'rm -rf -- "${stage_root}"' EXIT
bundle_root="${stage_root}/d435i_autonomous_vision_bundle"
mkdir -p "${bundle_root}/examples" \
  "${bundle_root}/ultralytics" \
  "${bundle_root}/runs/segment/drink_seg/weights" \
  "${bundle_root}/mesh" \
  "${bundle_root}/docs" \
  "${bundle_root}/licenses"

required_sources=(
  examples/d435i_autonomous_vision.py
  examples/d435i_foundationpose.py
  examples/async_pose_worker.py
  examples/multi_object_rgbd_tracker.py
  examples/rgbd_kalman_tracker.py
  examples/upright_pose_utils.py
  examples/depth_mask_utils.py
  examples/README_D435I_AUTONOMOUS_VISION_ZH.md
  runs/segment/drink_seg/weights/best.pt
  mesh/yinliao_edit_fixed.obj
  LICENSE
)
for relative_path in "${required_sources[@]}"; do
  [[ -f "${repo_root}/${relative_path}" ]] || {
    echo "ERROR: required source is missing: ${repo_root}/${relative_path}" >&2
    exit 1
  }
done

cp -a "${repo_root}/examples/d435i_autonomous_vision.py" "${bundle_root}/examples/"
cp -a "${repo_root}/examples/d435i_foundationpose.py" "${bundle_root}/examples/"
cp -a "${repo_root}/examples/async_pose_worker.py" "${bundle_root}/examples/"
cp -a "${repo_root}/examples/multi_object_rgbd_tracker.py" "${bundle_root}/examples/"
cp -a "${repo_root}/examples/rgbd_kalman_tracker.py" "${bundle_root}/examples/"
cp -a "${repo_root}/examples/upright_pose_utils.py" "${bundle_root}/examples/"
cp -a "${repo_root}/examples/depth_mask_utils.py" "${bundle_root}/examples/"
cp -a "${repo_root}/ultralytics/." "${bundle_root}/ultralytics/"
cp -a "${repo_root}/runs/segment/drink_seg/weights/best.pt" \
  "${bundle_root}/runs/segment/drink_seg/weights/"
cp -a "${repo_root}/mesh/yinliao_edit_fixed.obj" "${bundle_root}/mesh/"
cp -a "${repo_root}/examples/README_D435I_AUTONOMOUS_VISION_ZH.md" \
  "${bundle_root}/docs/"
cp -a "${repo_root}/LICENSE" "${bundle_root}/licenses/YOLOv8-GPL-3.0.txt"
cp -a "${packaging_dir}/README_PACKAGE_ZH.md" "${bundle_root}/README.md"
cp -a "${packaging_dir}/MANIFEST.txt" "${bundle_root}/"
cp -a "${packaging_dir}/requirements-app.txt" "${bundle_root}/"
cp -a "${packaging_dir}/THIRD_PARTY_NOTICES.md" "${bundle_root}/"
cp -a "${packaging_dir}/run_d435i.sh" "${bundle_root}/"
cp -a "${packaging_dir}/verify_install.py" "${bundle_root}/"
chmod +x "${bundle_root}/run_d435i.sh" "${bundle_root}/verify_install.py"

if [[ -n "${foundationpose_source}" ]]; then
  cp -a "${foundationpose_source}" "${bundle_root}/FoundationPose"
fi

tar \
  --exclude='*/.git' \
  --exclude='*/.vscode' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*/FoundationPose/assets' \
  --exclude='*/FoundationPose/mycpp/build' \
  -C "${stage_root}" \
  -czf "${output_path}" \
  d435i_autonomous_vision_bundle

(
  cd -- "$(dirname -- "${output_path}")"
  sha256sum "$(basename -- "${output_path}")" > "$(basename -- "${output_path}").sha256"
)

echo "Created: ${output_path}"
echo "Checksum: ${output_path}.sha256"
du -h "${output_path}"
