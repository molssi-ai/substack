#!/bin/bash
# Step 2: one GPU, one model. Every setting of both codes is timed on growing systems until it no
# longer fits in GPU memory; the last step is bisected. Appends to results/single/<model>.jsonl.
# SWEEP_ARGS adds options of bench/sweep.py (e.g. SWEEP_ARGS="--r-max 8" for a quick test).
#   sbatch jobs/single_gpu.sh mace      (from the package folder; or: bash jobs/single_gpu.sh mace)
#SBATCH --job-name=bench-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=bench-single_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source ./env.sh
MODEL="${1:?usage: single_gpu.sh mace|aimnet2|tensornet}"
case "$MODEL" in
  mace)      R0=5; VARIANTS="xnn:fast:f32 xnn:graph:f32 xnn:ref:f32 xnn:fast:f64 xnn:ref:f64
                             alchemi:cueq_compile:f32 alchemi:cueq:f32 alchemi:eager:f32 alchemi:eager:f64" ;;
  aimnet2)   R0=8; VARIANTS="xnn:fast:f32 xnn:ref:f32 xnn:fast:f64 xnn:ref:f64 alchemi:eager:f32 alchemi:compile:f32" ;;
  tensornet) R0=5; VARIANTS="xnn:fast:f32 xnn:ref:f32 xnn:fast:f64 xnn:ref:f64
                             alchemi:eager:f32 alchemi:compile:f32 alchemi:pyg:f32" ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
mkdir -p results/single
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# shellcheck disable=SC2086
"$PY" bench/sweep.py --model "$MODEL" --variants $VARIANTS --r0 "$R0" --out "results/single/$MODEL.jsonl" ${SWEEP_ARGS:-}
