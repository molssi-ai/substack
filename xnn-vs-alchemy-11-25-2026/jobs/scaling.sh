#!/bin/bash
# Step 3: eight GPUs of one node, one model. The article's system on 1, 2, 4 and 8 GPUs for both
# codes, then the system doubled on 8 GPUs until it no longer fits. Appends to results/scaling/<model>.jsonl.
# SCALING_ARGS adds options of bench/scaling.py (e.g. SCALING_ARGS="--worlds 1 2 --no-grow").
#   sbatch jobs/scaling.sh mace         (from the package folder; or: bash jobs/scaling.sh mace)
#SBATCH --job-name=bench-scaling
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=4:00:00
#SBATCH --output=bench-scaling_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source ./env.sh
MODEL="${1:?usage: scaling.sh mace|aimnet2|tensornet}"
mkdir -p results/scaling
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# shellcheck disable=SC2086
"$PY" bench/scaling.py --model "$MODEL" --out "results/scaling/$MODEL.jsonl" ${SCALING_ARGS:-}
