"""Time one point on one GPU and append it to a JSON-lines file.

    python point.py --model mace --side xnn --variant fast --dtype f32 --n 9000 --out mace.jsonl

Exit code 0: timed; 3: out of GPU memory; 1: any other failure (recorded too).
"""

import argparse
import json
import time
import traceback

import harness as H
import torch

VARIANTS = {"alchemi": ["eager", "compile", "cueq", "cueq_compile", "pyg"], "xnn": ["ref", "fast", "graph"]}


def build(args, device):
    dtype = H.DTYPES[args.dtype]
    sys_ = H.system(args.model, args.n)
    t0 = time.perf_counter()
    if args.side == "alchemi":
        step = H.alchemi_step(H.alchemi_model(args.model, args.variant, device, dtype), sys_, device, dtype)
        n_edges = None
    else:
        model = H.xnn_model(args.model, device, dtype, fast=args.variant != "ref")
        step, g = H.xnn_step(model, sys_, device, captured=args.variant == "graph")
        n_edges = int(g.edge_index.shape[1])
    torch.cuda.synchronize(device)
    return step, sys_, n_edges, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(H.ATOMS_PER_CELL))
    ap.add_argument("--side", required=True, choices=list(VARIANTS))
    ap.add_argument("--variant", required=True, help="alchemi: eager compile cueq cueq_compile pyg; "
                                                     "xnn: ref fast graph (captured)")
    ap.add_argument("--dtype", default="f32", choices=list(H.DTYPES))
    ap.add_argument("--n", type=int, required=True, help="atoms (rounded to whole cells)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.variant not in VARIANTS[args.side]:
        ap.error(f"variant {args.variant!r} is not one of {VARIANTS[args.side]}")
    device = torch.device("cuda")
    rec = {"model": args.model, "side": args.side, "variant": args.variant, "dtype": args.dtype,
           "n_request": args.n, "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__}
    code = 0
    try:
        step, sys_, n_edges, setup_s = build(args, device)
        rec.update(n_atoms=int(sys_.positions.shape[0]), n_edges=n_edges, setup_s=setup_s)
        t = H.time_step(step, device, args.iters, args.warmup)
        rec.update(ms=t.step_ms, peak_mb=t.peak_mem_mb, energy=t.final_energy_eV, status="ok")
    except Exception as exc:
        oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
        rec.update(status="oom" if oom else "error", error=f"{type(exc).__name__}: {str(exc)[:400]}")
        if not oom:
            traceback.print_exc()
        code = 3 if oom else 1
    with open(args.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
