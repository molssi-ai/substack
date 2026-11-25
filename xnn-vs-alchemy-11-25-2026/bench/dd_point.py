"""xnn's DomainParallel on one system over the GPUs of a torchrun launch; rank 0 appends a JSON line.

    torchrun --nproc-per-node=8 dd_point.py --model mace --n 995328 --out mace.jsonl [--check]

The partition (owned and ghost atoms, neighbor lists) is built by an untimed first call and reused
while the atoms stay within the skin, as in molecular dynamics. ``--check`` then evaluates the
whole system on one GPU and records the differences.
"""

import argparse
import dataclasses
import datetime
import json
import os
import time

import harness as H
import torch
import torch.distributed as dist


def whole_system(sys_, device):
    """The system as an xnn graph without edges: DomainParallel builds each GPU's own."""
    from xnn.common.data import AtomicGraph
    n = int(sys_.positions.shape[0])
    return AtomicGraph(pos=sys_.positions.to(device), atomic_numbers=sys_.atomic_numbers.to(device),
                       edge_index=torch.zeros((2, 0), dtype=torch.long, device=device),
                       cell_shifts=torch.zeros((0, 3), dtype=torch.long, device=device),
                       batch=torch.zeros(n, dtype=torch.long, device=device), n_atoms=torch.tensor([n], device=device),
                       cell=sys_.cell.to(device)[None], pbc=sys_.pbc.reshape(1, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(H.ATOMS_PER_CELL))
    ap.add_argument("--dtype", default="f32", choices=list(H.DTYPES))
    ap.add_argument("--n", type=int, required=True, help="atoms (rounded to whole cells)")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--check", action="store_true", help="compare with the whole system on one GPU")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=10))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    from xnn.common.deploy import DomainParallel

    sys_ = H.system(args.model, args.n)
    model = H.xnn_model(args.model, device, H.DTYPES[args.dtype], fast=True)
    g = whole_system(sys_, device)
    dp = DomainParallel(model)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    dp(dataclasses.replace(g))                  # builds and keeps the partition
    torch.cuda.synchronize(device)
    setup_s = time.perf_counter() - t0

    def step():
        out = dp(dataclasses.replace(g))
        return out["energy"], out["forces"], out["stress"]

    t = H.time_step(step, device, args.iters, args.warmup)
    plan = dp.plan(g)                           # the same partition, for its atom counts
    stats = torch.tensor([t.step_ms, t.peak_mem_mb, setup_s, plan.n_local, plan.n_owned], dtype=torch.float64,
                         device=device)
    gathered = [torch.zeros_like(stats) for _ in range(world)]
    dist.all_gather(gathered, stats)
    if rank == 0:
        rows = torch.stack(gathered).cpu()
        rec = {"model": args.model, "side": "xnn", "variant": "fast", "dtype": args.dtype, "world": world,
               "grid": list(plan.grid), "n_atoms": int(sys_.positions.shape[0]), "ms": float(rows[:, 0].max()),
               "peak_mb_max": float(rows[:, 1].max()), "setup_s": float(rows[:, 2].max()),
               "local_atoms": rows[:, 3].long().tolist(), "owned_atoms": rows[:, 4].long().tolist(),
               "energy": t.final_energy_eV, "gpu": torch.cuda.get_device_name(device), "status": "ok"}
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    if args.check:
        out = {k: v.detach() for k, v in dp(dataclasses.replace(g)).items()}
        del step
        torch.cuda.empty_cache()
        if rank == 0:
            try:
                whole = H.xnn_step(model, sys_, device)[0]()
                check = H.compare((out["energy"], out["forces"], out["stress"]), whole)
            except torch.OutOfMemoryError as exc:
                check = {"status": "oom", "error": str(exc)[:200]}
            with open(args.out, "a") as fh:
                fh.write(json.dumps({"model": args.model, "side": "xnn", "world": world,
                                     "n_atoms": int(sys_.positions.shape[0]), "check": check}) + "\n")
            print("check", json.dumps(check), flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
