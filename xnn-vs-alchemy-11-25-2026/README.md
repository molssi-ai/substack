# xnn vs NVIDIA ALCHEMI on B200 GPUs: reproduction package

This folder reproduces every number, table and figure of the article
*Accelerated Machine Learning Interatomic Potentials for Materials Discovery*.
It times three machine-learned interatomic potentials (MACE-MPA-0, AIMNet2 and
TensorNet) in two frameworks, **xnn** and **NVIDIA ALCHEMI** (nvalchemi-toolkit
0.3.0). Both codes load the same published weights and run NVIDIA's own
benchmark systems with NVIDIA's own timing loop.

In its comparison, the package uses three quantitative measures:

1. **One GPU.** The time of one step (energy, forces and stress of a periodic cell) at growing system
   sizes, up to the largest system that fits in GPU memory.
2. **Several GPUs.** One system split over 1, 2, 4 and 8 GPUs, then doubled on 8 GPUs until it no
   longer fits.
3. **Agreement.** Both codes give the same energies, forces and stresses from the same weights.

## Contents

```
README.md               this guide
install.sh              builds .venv and fetches NVIDIA's benchmark helpers into third_party/
requirements.txt        pinned package versions
env.sh                  sourced by the job scripts (Python, paths, compiler caches)
jobs/check.sh           step 1: agreement of the two codes (1 GPU, minutes)
jobs/single_gpu.sh      step 2: single-GPU sweeps of one model (1 GPU, hours)
jobs/scaling.sh         step 3: multi-GPU runs of one model (8 GPUs, 1 to 2 hours)
bench/                  the Python code the jobs run (see "The code" below)
reference_results/      the article's raw measurements, figures and tables
third_party/            NVIDIA's benchmark helpers, downloaded by install.sh (not included)
```

## Requirements

- **GPUs.** NVIDIA GPUs with a driver for CUDA 12.8 or newer. The article used one node with eight
  B200 GPUs (180 GB each). Steps 1 and 2 need one GPU; step 3 needs eight GPUs in one node. Any
  recent data-center GPU works. On GPUs with less memory, the largest systems are smaller, and the
  times differ from the B200's.
- **Software.** Linux x86_64, [uv](https://docs.astral.sh/uv/), `curl`, `tar` and `sha256sum`, and
  internet access for the installation and the first download of each model's weights. Slurm is
  optional: every job script also runs with `bash`.
- **Disk.** About 15 GB for the environment and the model weights.

## Step 0: install (once)

```bash
cd xnn-vs-alchemy-11-25-2026
bash install.sh
```

The script does four things:

- It builds `.venv` with PyTorch 2.10.0 (CUDA 12.8), nvalchemi-toolkit 0.3.0 with its MACE and
  AIMNet2 dependencies, MatGL 4.1.4, cuEquivariance 0.11, Warp 1.19.0 and xnn.
- It downloads NVIDIA's benchmark folder, `benchmark/distributed` of the nvalchemi-toolkit v0.3.0
  release (https://github.com/NVIDIA/nvalchemi-toolkit/tree/v0.3.0/benchmark/distributed), into
  `third_party/`. The release's Apache-2.0 `LICENSE` comes with it.
- It checks that the files the benchmark uses match the release byte for byte. A folder that does not
  match is downloaded again.
- It prints the installed versions.

NVIDIA's benchmark folder is not part of this package, because the job scripts use it unchanged from
NVIDIA's release. To download only that folder, run `bash install.sh benchmark`.

The default is xnn 0.28.0, the release the article used. `XNN_SPEC` selects another one, for example
`XNN_SPEC="xnns>=0.28" bash install.sh`, or a local source folder.

## Step 1: check that both codes agree (about 10 minutes, 1 GPU)

```bash
sbatch jobs/check.sh          # or: bash jobs/check.sh
```

Each line compares one model at one precision, for example:

```
mace       f64 xnn:ref   vs ALCHEMI:eager  energy rel 1.2e-15  forces max 3.8e-14 eV/A  stress rel 5.9e-15
```

Float64 differences sit at double-precision round-off (1e-15 to 1e-14). Float32 differences sit at
single-precision round-off (1e-6 to 1e-4). The results go to `results/check.jsonl`.

## Step 2: single-GPU sweeps (up to about 3 hours per model, 1 GPU)

```bash
sbatch jobs/single_gpu.sh mace
sbatch jobs/single_gpu.sh aimnet2
sbatch jobs/single_gpu.sh tensornet
```

Each job times every setting of both codes for one model:

| model | xnn settings | ALCHEMI settings |
|---|---|---|
| MACE | fast, captured (CUDA graph), reference; float32 and float64 | cuEquivariance + torch.compile, cuEquivariance, eager; eager float64 |
| AIMNet2 | fast, reference; float32 and float64 | eager, torch.compile (float32 only) |
| TensorNet | fast, reference; float32 and float64 | Warp eager, Warp + torch.compile, plain MatGL (PyG) |

Each setting starts small and doubles the number of atoms until a step runs out of GPU memory. It then
bisects between the last size that ran and the first that failed. Every point runs in its own
process, so a failure cannot affect the next point. The records are appended to
`results/single/<model>.jsonl`.

## Step 3: multi-GPU runs (1 to 2 hours per model, 8 GPUs)

```bash
sbatch jobs/scaling.sh mace
sbatch jobs/scaling.sh aimnet2
sbatch jobs/scaling.sh tensornet
```

Each job does two things:

- It runs the article's system (MACE and TensorNet: 124,416 atoms of quartz; AIMNet2: 787,320 atoms
  of methane) on 1, 2, 4 and 8 GPUs with both codes. On two or more GPUs, xnn's result is also
  compared with the whole system on one GPU.
- It then doubles the system on 8 GPUs until it no longer fits and bisects the last step.

ALCHEMI runs NVIDIA's own `benchmark_dd_model_forward.py`, unchanged, at its fastest float32 setting
(MACE: cuEquivariance + torch.compile). ALCHEMI has no multi-GPU mode for TensorNet, so only xnn runs
that model. The records are appended to `results/scaling/<model>.jsonl`.

Cluster options go on the `sbatch` command line, for example
`sbatch --account=myaccount --partition=gpu jobs/scaling.sh mace`.

## Step 4: figures and tables (seconds, no GPU)

```bash
source env.sh
"$PY" bench/figures.py results              # your measurements
"$PY" bench/figures.py reference_results    # the article's measurements
```

Each call writes six figures and a Markdown file of tables:

| file | content |
|---|---|
| `figures/fig1a_single_gpu_mace.png`, `fig1b_single_gpu_aimnet2.png`, `fig1c_single_gpu_tensornet.png` | time per step and speed-up on one GPU, one figure per model |
| `figures/fig2_largest_system.png` | largest system on one and eight GPUs |
| `figures/fig3_strong_scaling.png` | one system on 1 to 8 GPUs |
| `figures/fig4_eight_gpu_growth.png` | growing systems on eight GPUs |
| `tables.md` | every table of the article |

Put your `tables.md` next to `reference_results/tables.md` to compare the two runs.

## A quick test

Steps 2 and 3 accept extra options through `SWEEP_ARGS` and `SCALING_ARGS`. These options run the
whole pipeline on small systems in under an hour, with two GPUs:

```bash
SWEEP_ARGS="--r-max 5" bash jobs/single_gpu.sh mace
SCALING_ARGS="--worlds 1 2 --r 6 --r-max 8" bash jobs/scaling.sh mace
```

## What is measured

- **One step** evaluates the total energy, the forces on all atoms and the stress tensor of the whole
  periodic cell. That is the force call of one molecular-dynamics step.
- **Neighbor list.** Both codes build the neighbor list once, before timing.
- **Timing loop.** The loop is NVIDIA's own (`time_step` in `_benchmark_common.py`): 3 warm-up steps,
  then the mean wall-clock time of 10 timed steps (5 on several GPUs). The GPU is synchronized around
  each step.
- **Memory.** The peak memory is the most memory PyTorch allocated during the timed steps. The warm-up,
  and with it any compilation, is excluded from both time and memory.
- **Systems.** NVIDIA's builders (`build_sio2_supercell` and `build_methane_packing` in
  `_benchmark_common.py`) make them: alpha-quartz SiO2 repeated r x r x r times (9r³ atoms) for
  MACE and TensorNet, and a cubic lattice of methane molecules (5r³ atoms) for AIMNet2. NVIDIA's
  benchmark has no TensorNet configuration; TensorNet runs on the quartz system.
- **Weights.** Both codes load MACE-MPA-0 medium, the first member of the AIMNet2 wB97M-D3 ensemble,
  and TensorNet-MatPES-PBE-2025.2.
- **AIMNet2 energy.** ALCHEMI's AIMNet2 returns the network energy without AIMNet2's built-in
  short-range Coulomb term. The xnn side subtracts the same term (`NetworkMinusShortRange` in
  `bench/harness.py`).

## The code

| file | what it does |
|---|---|
| `bench/harness.py` | systems, both codes' models, step functions and timing, from NVIDIA's benchmark helpers |
| `bench/point.py` | times one (model, code, setting, precision, size) point on one GPU |
| `bench/sweep.py` | grows and bisects the system for each setting (calls `point.py`) |
| `bench/dd_point.py` | xnn's `DomainParallel` on one system, under `torchrun` |
| `bench/scaling.py` | multi-GPU runs of both codes (calls `dd_point.py` and NVIDIA's script) |
| `bench/check.py` | agreement of the two codes on small systems |
| `bench/figures.py` | figures and tables from a results folder |

Every script prints its options with `--help`. Each measurement is one JSON line. The main fields are:

- `model`, `side` (`xnn` or `alchemi`), `variant`, `dtype`;
- `n_atoms`, `world` (number of GPUs);
- `ms` (mean time per step) and `peak_mb` or `peak_mb_max` (peak memory, MiB);
- `energy`;
- `status` (`ok`, `oom`, `error` or `timeout`).

## Versions used for the article

PyTorch 2.10.0 (CUDA 12.8), Triton 3.6.0, Warp 1.19.0 (CUDA 12 build), cuEquivariance 0.11.1,
nvalchemi-toolkit 0.3.0, nvalchemi-toolkit-ops 0.5.0, mace-torch 0.3.15, aimnet 0.2.6, MatGL 4.1.4,
e3nn 0.4.4, vesin-torch 0.6.2, ASE 3.29.0, xnn 0.28.0, NVIDIA driver 595.91.07, 8 x NVIDIA B200.

## Troubleshooting

- **Warp fails to load on an older driver.** The PyPI wheel of Warp 1.19.0 needs CUDA 13. install.sh
  installs NVIDIA's CUDA 12 build instead; keep it.
- **A point shows `oom`.** That is how a sweep ends; it is expected. `error` or `timeout` points
  print a traceback in the job's output file.
- **Running a step again.** The scripts append to their results files. Delete `results/` first for a
  clean run.
- **Model downloads.** The first run downloads each model's weights to the usual caches (`~/.cache`).
  Set `XDG_CACHE_HOME` to move them.
- **cuEquivariance compiles its kernels on first use.** The first MACE point takes a few minutes
  longer. `env.sh` disables parallel compilation, which collides across GPUs.
