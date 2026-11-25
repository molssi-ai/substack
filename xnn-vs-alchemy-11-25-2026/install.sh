#!/bin/bash
# Builds the Python environment (.venv) and fetches NVIDIA's benchmark helpers (third_party/).
#   bash install.sh             everything
#   bash install.sh benchmark   only the benchmark helpers
# XNN_SPEC selects the xnn release (default below); VENV the environment directory.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HERE/.venv}"
XNN_SPEC="${XNN_SPEC:-xnns==0.28.0}"
TORCH_INDEX=https://download.pytorch.org/whl/cu128
WARP_WHEEL=https://github.com/NVIDIA/warp/releases/download/v1.19.0/warp_lang-1.19.0+cu12-py3-none-manylinux_2_28_x86_64.whl
NV_URL=https://github.com/NVIDIA/nvalchemi-toolkit/archive/refs/tags/v0.3.0.tar.gz

verify_benchmark() {
  # the files the benchmark uses, exactly as released
  (cd "$1" && sha256sum -c --quiet - <<'SUMS'
740b26cff01af0d42631a7ad6e9f8ac77cc6f1311c73f06bf2c3425ab160b996  _benchmark_common.py
7b23202276cd2cb01cb5aba082a2b101de8c618226188fc14b46e700b0c8401d  benchmark_dd_model_forward.py
d085e01571fe58e7f57337dc35c6ca011792b4474ebbfb2d5da0885d4f55b7db  configs/mace.yaml
65349a49d79a73330715569cd5e6db9b9a7392d8479dd5294606172f2be8639a  configs/aimnet2.yaml
SUMS
  )
}

fetch_benchmark() {
  # NVIDIA's own system builders, timing loop and multi-GPU benchmark (nvalchemi-toolkit v0.3.0 source,
  # Apache-2.0, with its LICENSE), downloaded and checked in a temporary folder, then moved into place
  local top="$HERE/third_party/nvalchemi-toolkit-0.3.0"
  if [ -f "$top/benchmark/distributed/_benchmark_common.py" ]; then
    verify_benchmark "$top/benchmark/distributed" && return 0
    echo "third_party/ does not match the v0.3.0 release; downloading it again"
    rm -rf "$top"
  fi
  local tool
  for tool in curl tar sha256sum; do
    command -v "$tool" >/dev/null || { echo "$tool is required to download NVIDIA's benchmark helpers"; exit 1; }
  done
  local tmp
  tmp="$(mktemp -d)"
  echo "downloading NVIDIA's benchmark helpers from $NV_URL"
  curl -fsSL "$NV_URL" -o "$tmp/nv.tar.gz"
  tar xzf "$tmp/nv.tar.gz" -C "$tmp" nvalchemi-toolkit-0.3.0/LICENSE nvalchemi-toolkit-0.3.0/benchmark/distributed
  verify_benchmark "$tmp/nvalchemi-toolkit-0.3.0/benchmark/distributed" || { echo "checksum mismatch"; rm -rf "$tmp"; exit 1; }
  mkdir -p "$HERE/third_party"
  mv "$tmp/nvalchemi-toolkit-0.3.0" "$top"
  rm -rf "$tmp"
  echo "benchmark helpers ready: $top/benchmark/distributed"
}

fetch_benchmark
[ "${1:-all}" = benchmark ] && exit 0

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
uv venv --python 3.12 "$VENV"
PY="$VENV/bin/python"
uv pip install -p "$PY" --index-url "$TORCH_INDEX" "torch==2.10.0+cu128" "torchvision==0.25.0+cu128"
printf 'torch==2.10.0+cu128\ntorchvision==0.25.0+cu128\n' > "$VENV/constraints.txt"
uv pip install -p "$PY" -c "$VENV/constraints.txt" --extra-index-url "$TORCH_INDEX" \
  --index-strategy unsafe-best-match -r "$HERE/requirements.txt" "$XNN_SPEC"
# Warp's CUDA 12 build: the PyPI wheel targets CUDA 13, which older drivers cannot run
uv pip install -p "$PY" --no-deps "$WARP_WHEEL"
"$PY" - <<'EOF'
import importlib
for name in ("torch", "triton", "warp", "nvalchemi", "mace", "aimnet", "matgl", "cuequivariance", "xnn"):
    mod = importlib.import_module(name)
    print(f"{name:16s} {getattr(mod, '__version__', '?')}")
EOF
echo "environment ready: $VENV"
