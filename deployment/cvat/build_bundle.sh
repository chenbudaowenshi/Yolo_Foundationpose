#!/usr/bin/env bash
set -euo pipefail

packaging_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${packaging_dir}/../.." && pwd)"
cvat_source="${repo_root}/tools/cvat"
output_path=""

usage() {
  cat <<'EOF'
Usage: build_bundle.sh [--output ARCHIVE.tar.gz]

Creates a small CVAT installer archive in dist/. Docker images are pulled on
the recipient machine; no datasets, annotations or local overrides are packed.
EOF
}

while (($#)); do
  case "$1" in
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

if [[ -z "${output_path}" ]]; then
  mkdir -p "${repo_root}/dist"
  output_path="${repo_root}/dist/cvat-annotation-bundle.tar.gz"
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

required_sources=(
  tools/cvat/docker-compose.yml
  tools/cvat/.env
  tools/cvat/LICENSE
  tools/cvat/components/analytics/grafana_conf.yml
  tools/cvat/components/analytics/vector/vector.toml
  tools/cvat/components/analytics/grafana/dashboards/all_events.json
  examples/prepare_cvat_segmentation.py
  examples/import_cvat_segmentation.py
)
for relative_path in "${required_sources[@]}"; do
  [[ -f "${repo_root}/${relative_path}" ]] || {
    echo "ERROR: required source is missing: ${repo_root}/${relative_path}" >&2
    exit 1
  }
done

stage_root="$(mktemp -d)"
trap 'rm -rf -- "${stage_root}"' EXIT
bundle_root="${stage_root}/cvat_annotation_bundle"
mkdir -p "${bundle_root}/cvat/components" "${bundle_root}/tools"

cp -a "${cvat_source}/docker-compose.yml" "${bundle_root}/cvat/"
cp -a "${cvat_source}/LICENSE" "${bundle_root}/cvat/LICENSE-CVAT-MIT.txt"
cp -a "${cvat_source}/components/analytics" "${bundle_root}/cvat/components/"
# Only the pinned version is carried over; install.sh writes CVAT_HOST.
grep '^CVAT_VERSION=' "${cvat_source}/.env" > "${bundle_root}/cvat/.env"
cp -a "${repo_root}/examples/prepare_cvat_segmentation.py" "${bundle_root}/tools/"
cp -a "${repo_root}/examples/import_cvat_segmentation.py" "${bundle_root}/tools/"
cp -a "${packaging_dir}/install.sh" "${bundle_root}/"
cp -a "${packaging_dir}/README_PACKAGE_ZH.md" "${bundle_root}/README.md"
chmod +x "${bundle_root}/install.sh"

tar \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -C "${stage_root}" \
  -czf "${output_path}" \
  cvat_annotation_bundle

(
  cd -- "$(dirname -- "${output_path}")"
  sha256sum "$(basename -- "${output_path}")" > "$(basename -- "${output_path}").sha256"
)

echo "Created: ${output_path}"
echo "Checksum: ${output_path}.sha256"
du -h "${output_path}"
