"""Figures and tables of the article from a results folder (no GPU needed).

    python figures.py reference_results        the published numbers
    python figures.py results                  your own run

Reads <dir>/single/<model>.jsonl (sweep.py) and <dir>/scaling/<model>.jsonl (scaling.py); writes
<dir>/figures/*.png and <dir>/tables.md.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

MODELS = {"mace": "MACE-MPA-0", "aimnet2": "AIMNet2", "tensornet": "TensorNet"}
QUARTZ = r"$\alpha$-quartz SiO$_2$"
SYSTEM = {"mace": QUARTZ, "aimnet2": r"methane, CH$_4$", "tensornet": QUARTZ}
# ALCHEMI's fastest settings (PyG TensorNet and eager MACE are slower everywhere)
FASTEST = {"mace": {"cueq_compile"}, "aimnet2": {"eager", "compile"}, "tensornet": {"eager", "compile"}}
XNN, ALC, IDEAL, INK, GRID = "#d62728", "#1b6e20", "#8a8984", "#000000", "#e4e3df"

plt.rcParams.update({
    "font.size": 11.5, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.labelsize": 12,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 200,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def read(path):
    return [json.loads(line) for line in open(path)] if path.exists() else []


def best(records, side, variants, dtype):
    """Fastest successful record per system size over the given variants."""
    out = {}
    for r in records:
        if (r.get("side"), r.get("dtype"), r.get("status")) == (side, dtype, "ok") and r.get("variant") in variants:
            if r["n_atoms"] not in out or r["ms"] < out[r["n_atoms"]]["ms"]:
                out[r["n_atoms"]] = r
    return dict(sorted(out.items()))


def xnn_single(records, dtype="f32"):
    """xnn's ordinary evaluation, or its captured replay (CUDA graph) where that is over 2 % faster."""
    out = best(records, "xnn", {"fast"}, dtype)
    for n, r in best(records, "xnn", {"graph"}, dtype).items():
        if n not in out or r["ms"] < 0.98 * out[n]["ms"]:
            out[n] = r
    return dict(sorted(out.items()))


def multi(records, side, world=None, n=None):
    return {(r["world"] if n is not None else r["n_atoms"]): r for r in records
            if r.get("side") == side and r.get("status") == "ok" and "check" not in r
            and (world is None or r["world"] == world) and (n is None or r["n_atoms"] == n)}


def atoms(n):
    return f"{n:,}"


def short(n):
    """Atom count with three significant figures: 876k, 1.64M."""
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def tick(v, _=None):
    return f"{v / 1e6:g}M" if v >= 1e6 else (f"{v / 1e3:g}k" if v >= 1e3 else f"{v:g}")


def log_axes(ax, gpus=False):
    ax.set_xscale("log", base=2 if gpus else 10)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    if not gpus:
        ax.xaxis.set_major_formatter(FuncFormatter(tick))
        ax.xaxis.set_minor_formatter(NullFormatter())


def line(ax, pts, color, mark_last=False):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=2, marker="o", ms=5, mec="white", mew=1, zorder=3)
    if mark_last:
        ax.plot(xs[-1], ys[-1], marker="o", ms=11, mfc="none", mec=color, mew=1.6, zorder=4)


def label_largest(ax, x, a, room, size=10):
    """Atom counts at the open circles: xnn's to the right of its last point, ALCHEMI's above and to
    the left (every curve rises with size, so both places are clear of the lines)."""
    nx = max(x)
    ax.annotate(f"{atoms(nx)}\natoms", (nx, x[nx]["ms"]), xytext=(10, 0), textcoords="offset points",
                ha="left", va="center", fontsize=size, linespacing=1.1)
    if a:
        na = max(a)
        ax.annotate(f"{atoms(na)}\natoms", (na, a[na]["ms"]), xytext=(-8, 9), textcoords="offset points",
                    ha="right", va="bottom", fontsize=size, linespacing=1.1)
    ax.set_xlim(ax.get_xlim()[0], nx * room)
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 2.2)


def titles(ax, model, extra="", note="", size=11):
    ax.set_title(f"{MODELS[model]}{extra}\n", loc="left")
    ax.text(0, 1.02, SYSTEM[model] + note, transform=ax.transAxes, fontsize=size, va="bottom")


def legend(fig, y, largest=True, ideal=False, size=11.5):
    style = dict(lw=2, marker="o", ms=5, mec="white")
    handles = [plt.Line2D([], [], color=XNN, label="xnn", **style),
               plt.Line2D([], [], color=ALC, label="ALCHEMI (fastest setting)", **style)]
    if largest:
        handles.append(plt.Line2D([], [], ls="none", marker="o", ms=11, mfc="none", mec=INK, mew=1.6,
                                  label="largest system that fits"))
    if ideal:
        handles.append(plt.Line2D([], [], color=IDEAL, lw=1.2, ls="--", label="ideal scaling of xnn"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, y), fontsize=size)


def save(fig, path, title, y, size=14):
    fig.suptitle(title, y=y, fontsize=size, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_single(S, out):
    """One figure per model: time per step (left) and xnn's speed-up at the same size (right)."""
    for letter, (model, (x, a)) in zip("abc", S.items()):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
        line(ax, [(n, r["ms"]) for n, r in x.items()], XNN, True)
        line(ax, [(n, r["ms"]) for n, r in a.items()], ALC, True)
        log_axes(ax)
        label_largest(ax, x, a, room=4.0)
        ax.set_title("Time per step", loc="left")
        ax.set_xlabel("atoms")
        ax.set_ylabel("time per step (ms)")
        speed = [(n, a[n]["ms"] / x[n]["ms"]) for n in sorted(set(x) & set(a))]
        bx.axhline(1, color=IDEAL, lw=1, ls="--", zorder=2)
        for n, v in speed:
            bx.bar(n / 1.16, v, width=n * 1.16 - n / 1.16, align="edge", color=XNN, zorder=3)
            bx.annotate(f"{v:.1f}", (n, v), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                        fontsize=10)
        bx.set_xscale("log")
        bx.xaxis.set_major_formatter(FuncFormatter(tick))
        bx.xaxis.set_minor_formatter(NullFormatter())
        bx.set_ylim(0, max(v for _, v in speed) * 1.18)
        bx.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
        bx.grid(axis="x", visible=False)
        bx.set_title("xnn speed-up at the same size", loc="left")
        bx.set_xlabel("atoms")
        bx.set_ylabel("ALCHEMI time / xnn time")
        legend(fig, 1.02)
        save(fig, out / f"fig1{letter}_single_gpu_{model}.png",
             f"{MODELS[model]} on {SYSTEM[model]}, one B200 GPU", 1.1)


def largest_rows(S, S64, M):
    rows = [(f"{MODELS[m]}, 1 GPU", max(x), max(a)) for m, (x, a) in S.items()]
    rows.append(("MACE-MPA-0 float64, 1 GPU", max(S64[0]), max(S64[1])))
    for m, (recs, _) in M.items():
        xo, ao = multi(recs, "xnn", world=8), multi(recs, "alchemi", world=8)
        if xo:
            rows.append((f"{MODELS[m]}, 8 GPUs", max(xo), max(ao) if ao else None))
    return rows


def fig_largest(rows, out):
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    h = 0.38
    for i, (_, nx, na) in enumerate(rows):
        ax.barh(i + h / 2 + 0.02, nx, h, color=XNN, zorder=3)
        ax.text(nx * 1.1, i + h / 2 + 0.02, short(nx), va="center", fontsize=10)
        if na:
            ax.barh(i - h / 2 - 0.02, na, h, color=ALC, zorder=3)
            ax.text(na * 1.1, i - h / 2 - 0.02, f"{short(na)}   (xnn {nx / na:.1f}x larger)", va="center", fontsize=10)
        else:
            ax.text(1.15e3, i - h / 2 - 0.02, "ALCHEMI: no multi-GPU support", va="center", fontsize=10)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(-0.8, len(rows) - 0.4)
    ax.set_xscale("log")
    ax.set_xlim(1e3, 6e7)
    ax.xaxis.set_major_formatter(FuncFormatter(tick))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("atoms in the largest system that fits (log scale)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=XNN, label="xnn"),
                       plt.Rectangle((0, 0), 1, 1, color=ALC, label="ALCHEMI")],
              loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=2, fontsize=11.5)
    ax.set_title("Largest system that fits in GPU memory (B200, 180 GB per GPU)", loc="left", pad=28)
    fig.tight_layout()
    fig.savefig(out / "fig2_largest_system.png", bbox_inches="tight")
    plt.close(fig)


# larger text for the three-panel multi-GPU figures
LARGE = {"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 15, "xtick.labelsize": 13.5, "ytick.labelsize": 13.5}


@plt.rc_context(LARGE)
def fig_strong(M, out):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.3))
    for ax, (m, (recs, n0)) in zip(axes, M.items()):
        x, a = multi(recs, "xnn", n=n0), multi(recs, "alchemi", n=n0)
        w = sorted(x)
        if 1 not in x:
            continue
        ax.plot(w, [x[1]["ms"] / k for k in w], color=IDEAL, lw=1.2, ls="--", zorder=2)
        line(ax, [(k, x[k]["ms"]) for k in w], XNN)
        if a:
            line(ax, [(k, a[k]["ms"]) for k in sorted(a)], ALC)
        log_axes(ax, gpus=True)
        ax.set_xticks(w, [str(k) for k in w])
        ax.margins(y=0.1)
        titles(ax, m, f", {atoms(n0)} atoms", "" if a else ", xnn only", size=13.5)
        ax.set_xlabel("number of GPUs")
        if ax is axes[0]:
            ax.set_ylabel("time per step (ms)")
    legend(fig, 1.02, largest=False, ideal=True, size=14)
    save(fig, out / "fig3_strong_scaling.png", "Strong scaling: one system split over 1 to 8 B200 GPUs (float32)", 1.1,
         size=16)


@plt.rc_context(LARGE)
def fig_growth(M, out):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.3))
    for ax, (m, (recs, _)) in zip(axes, M.items()):
        x, a = (dict(sorted(multi(recs, s, world=8).items())) for s in ("xnn", "alchemi"))
        if not x:
            continue
        line(ax, [(n, r["ms"]) for n, r in x.items()], XNN, True)
        if a:
            line(ax, [(n, r["ms"]) for n, r in a.items()], ALC, True)
        log_axes(ax)
        ax.xaxis.set_major_locator(LogLocator(10, subs=(1, 3)))
        ax.margins(x=0.08, y=0.1)
        label_largest(ax, x, a, room=5.0, size=12)
        titles(ax, m, note="" if a else ", xnn only", size=13.5)
        ax.set_xlabel("atoms")
        if ax is axes[0]:
            ax.set_ylabel("time per step on 8 GPUs (ms)")
    legend(fig, 1.02, size=14)
    save(fig, out / "fig4_eight_gpu_growth.png",
         "Eight B200 GPUs: growing the system until it no longer fits (float32)", 1.1, size=16)


def table(title, header, rows):
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def gb(mb):
    return f"{mb / 1024:.1f}" if mb else ""


def tables(S, S64, singles, M, rows_largest):
    md = ["# Tables (generated by figures.py)\n"]
    fast = {m: best(singles[m], "xnn", {"fast"}, "f32") for m in S}
    for m, (x, a), f64 in [(m, xa, False) for m, xa in S.items()] + [("mace", S64, True)]:
        rows = []
        for n in sorted(set(x) | set(a)):
            xr, ar = x.get(n), a.get(n)
            xs = (f"{xr['ms']:.1f}" + (" (captured)" if xr["variant"] == "graph" else "")) if xr else \
                ("out of memory" if n > max(x) else "")
            as_ = f"{ar['ms']:.1f} ({ar['variant']})" if ar else ("out of memory" if n > max(a) else "")
            mem = fast[m].get(n, xr) if xr and xr["variant"] == "graph" else xr
            rows.append((atoms(n), xs, as_, f"{ar['ms'] / xr['ms']:.2f}x" if xr and ar else "",
                         gb(mem["peak_mb"]) if mem else "", gb(ar["peak_mb"]) if ar else ""))
        md.append(table(f"{MODELS[m]} on one GPU, {'float64' if f64 else 'float32'}, ms per step",
                        ["atoms", "xnn", "ALCHEMI", "speed-up", "xnn GB", "ALCHEMI GB"], rows))
    md.append(table("Largest system that fits", ["case", "xnn", "ALCHEMI", "ratio"],
                    [(c, atoms(nx), atoms(na) if na else "n/a", f"{nx / na:.1f}x" if na else "")
                     for c, nx, na in rows_largest]))
    rows = []
    for m, (recs, n0) in M.items():
        x, a = multi(recs, "xnn", n=n0), multi(recs, "alchemi", n=n0)
        for k in sorted(x):
            rows.append((f"{MODELS[m]}, {atoms(n0)}", k, f"{x[k]['ms']:.1f}", f"{a[k]['ms']:.1f}" if k in a else "",
                         f"{a[k]['ms'] / x[k]['ms']:.2f}x" if k in a else ""))
        if 1 in x and 8 in x:
            eff_a = f"{a[1]['ms'] / (8 * a[8]['ms']):.2f}" if 1 in a and 8 in a else ""
            rows.append((f"{MODELS[m]}, efficiency on 8", "", f"{x[1]['ms'] / (8 * x[8]['ms']):.2f}", eff_a, ""))
    md.append(table("Strong scaling, ms per step", ["model, atoms", "GPUs", "xnn", "ALCHEMI", "speed-up"], rows))
    rows = []
    for m, (recs, _) in M.items():
        x, a = multi(recs, "xnn", world=8), multi(recs, "alchemi", world=8)
        for n in sorted(set(x) | set(a)):
            rows.append((MODELS[m], atoms(n), f"{x[n]['ms']:.1f}" if n in x else "",
                         f"{a[n]['ms']:.1f}" if n in a else ("out of memory" if a and n > max(a) else ""),
                         gb(x[n].get("peak_mb_max")) if n in x else ""))
    md.append(table("Eight GPUs, ms per step", ["model", "atoms", "xnn", "ALCHEMI", "xnn GB per GPU"], rows))
    checks = [(MODELS[m], r["world"], atoms(r["n_atoms"]), f"{r['check']['energy']['rel']:.1e}",
               f"{r['check']['forces']['max_abs']:.1e}", f"{r['check']['stress']['rel']:.1e}")
              for m, (recs, _) in M.items() for r in recs if "energy" in r.get("check", {})]
    md.append(table("Several GPUs against the whole system on one GPU (xnn)",
                    ["model", "GPUs", "atoms", "energy rel", "forces max (eV/A)", "stress rel"], checks))
    rows = []
    for m in S:
        recs = singles[m]
        ref = {r["n_atoms"]: r["energy"] for r in recs if r.get("status") == "ok" and r["side"] == "xnn"
               and r["variant"] == "fast" and r["dtype"] == "f64"}

        def worst(side, variants, recs=recs, ref=ref):
            e = [(abs(r["energy"] - ref[r["n_atoms"]]), r["n_atoms"]) for r in recs if r.get("status") == "ok"
                 and r["side"] == side and r["variant"] in variants and r["dtype"] == "f32" and r["n_atoms"] in ref]
            return max(e, default=None, key=lambda t: t[0] / abs(ref[t[1]]))

        wx, wa = worst("xnn", {"fast"}), worst("alchemi", FASTEST[m])
        if wx and wa:
            rows.append((MODELS[m], f"{wx[0]:.2g} eV at {atoms(wx[1])}", f"{wa[0]:.3g} eV at {atoms(wa[1])}"))
    md.append(table("Float32 total energy against float64, same structure (worst over the sweep)",
                    ["model", "xnn", "ALCHEMI"], rows))
    rows = []
    for m, (x, a) in S.items():
        common = sorted(set(x) & set(a))
        if common:
            n = common[-1]
            rows.append((f"{MODELS[m]}, {atoms(n)} atoms", 1, f"{86.4 / x[n]['ms']:.3g} ns",
                         f"{86.4 / a[n]['ms']:.3g} ns"))
        recs, n0 = M[m]
        xs, as_ = multi(recs, "xnn", n=n0), multi(recs, "alchemi", n=n0)
        if 8 in xs:
            rows.append((f"{MODELS[m]}, {atoms(n0)} atoms", 8, f"{86.4 / xs[8]['ms']:.3g} ns",
                         f"{86.4 / as_[8]['ms']:.3g} ns" if 8 in as_ else "n/a"))
    md.append(table("Simulated time per day at a 1 fs time step (force calls only)",
                    ["system", "GPUs", "xnn", "ALCHEMI"], rows))
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    args = ap.parse_args()
    singles = {m: read(args.results / "single" / f"{m}.jsonl") for m in MODELS}
    scaling = {m: read(args.results / "scaling" / f"{m}.jsonl") for m in MODELS}
    S = {m: (xnn_single(singles[m]), best(singles[m], "alchemi", FASTEST[m], "f32")) for m in MODELS}
    S64 = (best(singles["mace"], "xnn", {"fast"}, "f64"), best(singles["mace"], "alchemi", {"eager"}, "f64"))
    # the strong-scaling system of each model: the one run on a single GPU by the multi-GPU script
    M = {m: (scaling[m], min(multi(scaling[m], "xnn", world=1))) for m in MODELS if multi(scaling[m], "xnn", world=1)}
    out = args.results / "figures"
    out.mkdir(exist_ok=True)
    rows_largest = largest_rows(S, S64, M)
    fig_single(S, out)
    fig_largest(rows_largest, out)
    fig_strong(M, out)
    fig_growth(M, out)
    (args.results / "tables.md").write_text(tables(S, S64, singles, M, rows_largest))
    print(f"wrote {out}/*.png and {args.results / 'tables.md'}")


if __name__ == "__main__":
    main()
