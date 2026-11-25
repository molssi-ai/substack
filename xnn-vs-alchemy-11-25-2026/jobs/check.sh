#!/bin/bash
# Step 1: do both codes give the same energies, forces and stresses from the same weights? A few
# minutes on one GPU; writes results/check.jsonl and prints one line per case.
#   sbatch jobs/check.sh                (from the package folder; or: bash jobs/check.sh)
#SBATCH --job-name=bench-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=bench-check_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source ./env.sh
mkdir -p results
"$PY" bench/check.py --out results/check.jsonl
