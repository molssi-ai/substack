# Sourced by every job script: the Python of .venv, NVIDIA's benchmark helpers, compiler caches.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$ROOT/.venv}"
PY="${PY:-$VENV/bin/python}"
export NV_BENCH="$ROOT/third_party/nvalchemi-toolkit-0.3.0/benchmark/distributed"
export PYTHONPATH="$ROOT/bench${PYTHONPATH:+:$PYTHONPATH}"
# CUDA libraries shipped as wheels (cuEquivariance, NCCL)
for d in "$VENV"/lib/python3.*/site-packages/nvidia/*/lib; do
  [ -d "$d" ] && LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done
export LD_LIBRARY_PATH
CACHE="${BENCH_CACHE:-$ROOT/.cache}"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" TRITON_CACHE_DIR="$CACHE/triton" WARP_CACHE_PATH="$CACHE/warp"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1 MPLCONFIGDIR="$CACHE/mpl"
# cuEquivariance compiles its kernels once per process; parallel compiles of several ranks collide
export CUEQUIVARIANCE_OPS_PARALLEL_COMPILE=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
[ -f "$NV_BENCH/_benchmark_common.py" ] || { echo "run first: bash $ROOT/install.sh"; return 1 2>/dev/null || exit 1; }
