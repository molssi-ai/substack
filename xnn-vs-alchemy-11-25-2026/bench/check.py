"""Do both codes compute the same energies, forces and stresses from the same weights?

Each case evaluates one small system on both sides and appends the largest differences to a
JSON-lines file. Float64 cases compare the implementations exactly; float32 cases show round-off.

    python check.py --out check.jsonl
"""

import argparse
import json

import harness as H
import torch

# model, xnn variant, ALCHEMI variant, precision, atoms
CASES = [("mace", "ref", "eager", "f64", 1125), ("mace", "fast", "cueq", "f32", 1125),
         ("aimnet2", "fast", "eager", "f32", 625), ("tensornet", "ref", "pyg", "f64", 1125),
         ("tensornet", "fast", "eager", "f32", 1125)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    for model, xv, av, dt, n in CASES:
        dtype = H.DTYPES[dt]
        sys_ = H.system(model, n)
        rec = {"model": model, "xnn": xv, "alchemi": av, "dtype": dt, "n_atoms": int(sys_.positions.shape[0])}
        try:
            ours = H.xnn_step(H.xnn_model(model, device, dtype, fast=xv == "fast"), sys_, device)[0]()
            theirs = H.alchemi_step(H.alchemi_model(model, av, device, dtype), sys_, device, dtype)()
            rec.update(diff=H.compare(ours, theirs), status="ok")
        except Exception as exc:
            rec.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        d = rec.get("diff")
        print(f"{model:10s} {dt} xnn:{xv:5s} vs ALCHEMI:{av:6s} " +
              (f"energy rel {d['energy']['rel']:.1e}  forces max {d['forces']['max_abs']:.1e} eV/A  "
               f"stress rel {d['stress']['rel']:.1e}" if d else rec["error"]), flush=True)


if __name__ == "__main__":
    main()
