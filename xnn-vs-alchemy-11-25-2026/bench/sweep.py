"""Grow the system of each variant until it no longer fits on the GPU, then bisect the largest size.

Sizes are whole cells repeated r times along each axis; each growth step multiplies r by 2^(1/3),
doubling the atoms. Every point runs in its own process (point.py), so a failure cannot leak.

    python sweep.py --model mace --variants xnn:fast:f32 alchemi:cueq_compile:f32 --r0 5 --out mace.jsonl
"""

import argparse
import json
import os
import subprocess
import sys

from harness import atoms

HERE = os.path.dirname(os.path.abspath(__file__))


def run(model, side, variant, dtype, r, args):
    n = atoms(model, r)
    cmd = [sys.executable, os.path.join(HERE, "point.py"), "--model", model, "--side", side, "--variant", variant,
           "--dtype", dtype, "--n", str(n), "--iters", str(args.iters), "--warmup", str(args.warmup),
           "--out", args.out]
    try:
        return {0: "ok", 3: "oom"}.get(subprocess.run(cmd, timeout=args.point_timeout).returncode, "error")
    except subprocess.TimeoutExpired:
        with open(args.out, "a") as fh:
            fh.write(json.dumps({"model": model, "side": side, "variant": variant, "dtype": dtype,
                                 "n_request": n, "status": "timeout"}) + "\n")
        return "timeout"


def sweep(model, side, variant, dtype, args):
    r, last_ok, first_bad = args.r0, None, None
    while r <= args.r_max:
        if run(model, side, variant, dtype, r, args) != "ok":
            first_bad = r
            break
        last_ok, r = r, max(r + 1, round(r * 2 ** (1 / 3)))
    if args.bisect and last_ok is not None and first_bad is not None:
        lo, hi = last_ok, first_bad
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if run(model, side, variant, dtype, mid, args) == "ok" else (lo, mid)
        last_ok = lo
    if last_ok is not None:
        print(f"=== {model} {side}:{variant}:{dtype} largest system {atoms(model, last_ok)} atoms", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", nargs="+", required=True, help="side:variant:dtype, e.g. xnn:fast:f32")
    ap.add_argument("--r0", type=int, default=5, help="repeats of the first system")
    ap.add_argument("--r-max", type=int, default=400, help="stop growing past these repeats")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--point-timeout", type=int, default=3600, help="seconds per point")
    ap.add_argument("--no-bisect", dest="bisect", action="store_false")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    for spec in args.variants:
        sweep(args.model, *spec.split(":"), args)


if __name__ == "__main__":
    main()
