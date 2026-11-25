"""Multi-GPU runs of both codes: one system over 1, 2, 4, 8 GPUs, then the system doubled on all GPUs
until it no longer fits (and the last step bisected).

xnn: dd_point.py (DomainParallel) under torchrun. ALCHEMI: NVIDIA's own
benchmark_dd_model_forward.py (nvalchemi-toolkit v0.3.0) under torchrun, with its printed table parsed;
on one GPU it runs its single-GPU path. ALCHEMI has no multi-GPU mode for TensorNet.

    python scaling.py --model mace --out mace.jsonl
"""

import argparse
import json
import os
import re
import subprocess
import sys

from harness import atoms

HERE = os.path.dirname(os.path.abspath(__file__))
# the strong-scaling system: the largest one both codes fit on one B200 (repeats per axis)
STRONG_R = {"mace": 24, "aimnet2": 54, "tensornet": 24}
# ALCHEMI's fastest float32 setting per model
NV_SETTINGS = {"mace": ["dtype=fp32", "loader.enable_cueq=true", "loader.compile=true"], "aimnet2": ["dtype=fp32"]}
NV_CONFIG = {"mace": "mace.yaml", "aimnet2": "aimnet2.yaml"}
ROW = re.compile(r"^\S+\s+\d+\s+.*$")


def torchrun(nproc):
    return [sys.executable, "-m", "torch.distributed.run", f"--nproc-per-node={nproc}",
            "--rdzv-backend", "c10d", "--rdzv-endpoint", "localhost:0"]


def write(path, rec):
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def run_xnn(model, n, world, args, check=False):
    cmd = torchrun(world) + [os.path.join(HERE, "dd_point.py"), "--model", model, "--n", str(n), "--iters",
                             str(args.iters), "--warmup", str(args.warmup), "--out", args.out]
    p = subprocess.run(cmd + (["--check"] if check else []), timeout=args.timeout, capture_output=True, text=True)
    sys.stdout.write(p.stdout[-2000:])
    if p.returncode != 0:
        tail = (p.stdout + p.stderr)[-1500:]
        write(args.out, {"model": model, "side": "xnn", "world": world, "n_request": n,
                         "status": "oom" if "out of memory" in tail.lower() else "error", "tail": tail})
        return False
    return True


def run_alchemi(model, n, world, args, check=False):
    script = os.path.join(os.environ["NV_BENCH"], "benchmark_dd_model_forward.py")
    cmd = ["--config", os.path.join(os.environ["NV_BENCH"], "configs", NV_CONFIG[model]), "--sizes", str(n),
           "--iters", str(args.iters), "--warmup", str(args.warmup)]
    for s in NV_SETTINGS[model]:
        cmd += ["--set", s]
    cmd = [sys.executable, script] + cmd + ["--single-only"] if world == 1 else torchrun(world) + [script] + cmd
    p = subprocess.run(cmd, timeout=args.timeout, capture_output=True, text=True)
    rows = [line for line in p.stdout.splitlines() if ROW.match(line.strip()) and " ms" in line]
    rec = {"model": model, "side": "alchemi", "world": world, "n_request": n, "settings": NV_SETTINGS[model],
           "rows": rows}
    ms = None
    for line in rows:
        times = [float(x) for x in re.findall(r"([0-9.]+) ms", line)]
        rec["n_atoms"] = int(line.split()[1])
        # one GPU: the single-GPU column; several: the decomposed column after it
        if world == 1 and times:
            ms = times[0]
        elif len(times) >= 2:
            ms = times[1]
    text = p.stdout + p.stderr
    if ms and ms > 0 and "OOM" not in p.stdout:
        rec.update(ms=ms, status="ok")
    else:
        rec.update(status="oom" if "OOM" in p.stdout or "out of memory" in text.lower() else "error",
                   tail=text[-1500:])
    write(args.out, rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "tail"}), flush=True)
    return rec["status"] == "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(STRONG_R))
    ap.add_argument("--r", type=int, help="repeats of the strong-scaling system (default: the article's)")
    ap.add_argument("--worlds", type=int, nargs="+", default=[1, 2, 4, 8], help="GPU counts")
    ap.add_argument("--sides", nargs="+", default=["xnn", "alchemi"])
    ap.add_argument("--no-grow", dest="grow", action="store_false", help="strong scaling only")
    ap.add_argument("--r-max", type=int, default=10 ** 6, help="stop growing past these repeats")
    ap.add_argument("--refine", type=int, default=4, help="bisection steps after the first failure")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per launch")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    r0 = args.r or STRONG_R[args.model]
    runner = {"xnn": run_xnn, "alchemi": run_alchemi}
    for side in args.sides:
        if side == "alchemi" and args.model not in NV_CONFIG:
            print(f"ALCHEMI has no multi-GPU mode for {args.model}", flush=True)
            continue
        for world in args.worlds:
            runner[side](args.model, atoms(args.model, r0), world, args, check=side == "xnn" and world > 1)
        if not args.grow:
            continue
        r, world = r0, max(args.worlds)
        while True:
            last, r = r, max(r + 1, round(r * 2 ** (1 / 3)))
            if r > args.r_max:
                break
            if not runner[side](args.model, atoms(args.model, r), world, args):
                lo, hi = last, r
                for _ in range(args.refine):
                    mid = (lo + hi) // 2
                    if mid <= lo:
                        break
                    lo, hi = (mid, hi) if runner[side](args.model, atoms(args.model, mid), world, args) else (lo, mid)
                break


if __name__ == "__main__":
    main()
