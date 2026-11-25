"""Both codes on NVIDIA's own benchmark systems, with NVIDIA's own timing loop.

The systems, ALCHEMI's model loaders and the timing loop come from the benchmark folder of the
nvalchemi-toolkit v0.3.0 source (``NV_BENCH``, fetched by install.sh): alpha-quartz SiO2 for MACE
and TensorNet, a periodic packing of methane for AIMNet2. One step is the energy, the forces and
the stress of the whole periodic cell. The neighbor list is built once, before timing, on both sides.
"""

import dataclasses
import os
import sys
import warnings

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.environ["NV_BENCH"])
import _benchmark_common as nvb  # noqa: E402

DTYPES = {"f32": torch.float32, "f64": torch.float64}
ATOMS_PER_CELL = {"mace": 9, "tensornet": 9, "aimnet2": 5}       # quartz unit cell, one CH4 molecule
NV_CONFIG = {"mace": "mace.yaml", "aimnet2": "aimnet2.yaml", "tensornet": "mace.yaml"}
MATGL_TENSORNET = "TensorNet-PES-MatPES-PBE-2025.2"


def atoms(model, r):
    """Atoms of the cell repeated ``r`` times along each axis."""
    return ATOMS_PER_CELL[model] * r ** 3


def nv_config(model, overrides=()):
    return nvb.load_config(os.path.join(os.environ["NV_BENCH"], "configs", NV_CONFIG[model]), list(overrides))


def system(model, n_atoms):
    """NVIDIA's test system of a model with about ``n_atoms`` atoms (float64, CPU)."""
    return nvb.build_system(nv_config(model), n_atoms, torch.float64)


def time_step(step, device, iters, warmup):
    """NVIDIA's timing loop: mean wall-clock ms of ``iters`` steps after ``warmup`` steps, and the
    peak memory allocated during the timed steps."""
    return nvb.time_step(step, device, n_iters=iters, n_warmup=warmup)


def matgl_tensornet(warp):
    """MatGL's TensorNet-PES-MatPES-PBE-2025.2. The published checkpoint runs plain PyTorch
    (``use_warp=False``); with ``warp`` the same weights go into MatGL's Warp-kernel network,
    the path ALCHEMI uses."""
    import matgl
    from matgl.models import TensorNet
    pot = matgl.load_model(MATGL_TENSORNET)
    if warp:
        old = pot.model
        new = TensorNet(element_types=old.element_types, units=128, nblocks=2, num_rbf=32, max_n=3, max_l=3,
                        rbf_type="SphericalBessel", use_smooth=True, activation_type="swish", cutoff=5.0,
                        equivariance_invariance_group="O(3)", is_intensive=False, readout_type="weighted_atom",
                        use_warp=True)
        new.load_state_dict(old.state_dict(), strict=True)
        pot.model = new
    return pot


def alchemi_model(model, variant, device, dtype):
    """ALCHEMI's wrapper of a model. Variants: ``eager``, ``compile`` (torch.compile), ``cueq`` and
    ``cueq_compile`` (MACE on cuEquivariance), ``pyg`` (TensorNet as published, without Warp)."""
    compile_model = "compile" in variant
    if model == "tensornet":
        from matgl.ext.alchmtk import TensorNetWrapper
        wrapper = TensorNetWrapper.from_potential(matgl_tensornet(warp=variant != "pyg")).to(device=device, dtype=dtype)
        if compile_model:
            wrapper.model = torch.compile(wrapper.model)
        return wrapper.eval()
    overrides = [f"loader.enable_cueq={str('cueq' in variant).lower()}"] if model == "mace" else []
    return nvb.build_loader(nv_config(model, overrides), compile_model=compile_model)(device, dtype)


def alchemi_step(wrapper, sys_, device, dtype):
    """A ``step() -> (energy, forces, stress)`` closure, built as NVIDIA's benchmark builds it."""
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.neighbors import compute_neighbors
    config = wrapper.model_config
    config.active_outputs = set(config.active_outputs) | {"stress"}
    fields = dict(atomic_numbers=sys_.atomic_numbers.to(device),
                  positions=sys_.positions.to(device=device, dtype=dtype).clone(),
                  atomic_masses=sys_.masses.to(device=device, dtype=dtype),
                  cell=sys_.cell.to(device=device, dtype=dtype).unsqueeze(0),
                  pbc=sys_.pbc.reshape(1, 3).to(device))
    if "aimnet" in type(wrapper).__name__.lower():
        fields["charge"] = torch.zeros(1, 1, dtype=dtype, device=device)
    batch = Batch.from_data_list([AtomicData(**fields)], device=device)
    compute_neighbors(batch, config=config.neighbor_config)

    def step():
        out = wrapper(batch)
        return out["energy"].sum(), out["forces"], out["stress"]
    return step


class NetworkMinusShortRange(torch.nn.Module):
    """AIMNet2's network energy minus its built-in short-range Coulomb term, the quantity ALCHEMI's
    AIMNet2 returns (ALCHEMI adds electrostatics separately, by Ewald or PME)."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.cutoff = model.cutoff

    def receptive_field(self):
        return self.model.receptive_field()

    @property
    def refreshes_ghosts(self):
        # the wrapped model's ghost mode: a one-cutoff halo refreshed before every layer
        return getattr(self.model, "refreshes_ghosts", False)

    def forward(self, data):
        from xnn.common.models.ops import gather_rows, scatter_sum, structure_sum
        m = self.model
        out = m(data)
        q = out["charges"]
        r = torch.linalg.norm(data.edge_vectors(), dim=-1)
        src, dst = data.edge_index[0], data.edge_index[1]
        pair = 0.5 * m.coulomb_constant * m.sr_envelope(r) * gather_rows(q, src) * gather_rows(q, dst) / r
        out["node_energy"] = out["node_energy"] - scatter_sum(pair.to(torch.float64), dst, data.num_nodes)
        out["energy"] = structure_sum(out["node_energy"], data.batch, data.num_graphs)
        return out


def xnn_model(model, device, dtype, fast):
    """xnn's model with the same published weights. ``fast`` selects the fused kernels, otherwise
    the reference implementation; both compute the same numbers."""
    from xnn.gnn.models import MACE, AIMNet2, TensorNet
    if model == "mace":
        m = MACE.from_foundation("mace-mpa-0-medium", dtype=dtype)
    elif model == "aimnet2":
        m = AIMNet2.from_foundation("aimnet2", dtype=dtype, dispersion=False, coulomb=None)
    else:
        m = TensorNet.from_foundation("tensornet-matpes-pbe-2025.2", dtype=dtype)
    m = m.to(device).eval()
    m.set_use_fast(fast)
    return NetworkMinusShortRange(m).eval() if model == "aimnet2" else m


def xnn_graph(model, sys_, device):
    """The whole cell as one xnn graph with its neighbor list (float64 geometry)."""
    from xnn.common.data.dataset import structure_to_graph
    return structure_to_graph({"pos": sys_.positions, "atomic_numbers": sys_.atomic_numbers,
                               "cell": sys_.cell, "pbc": sys_.pbc}, float(model.cutoff), device=device)


def xnn_step(model, sys_, device, captured=False):
    """A ``step() -> (energy, forces, stress)`` closure; ``captured`` replays it as a CUDA graph
    where that is faster (xnn's ``cuda_graph=True``)."""
    from xnn.common.models.outputs import ForceStressOutput
    wrapped = ForceStressOutput(model, compute_forces=True, compute_stress=True).eval()
    g = xnn_graph(model, sys_, device)
    if captured:
        from xnn.common.deploy import GraphedEvaluation
        for _ in range(5):                  # TorchScript blocks finish compiling before the capture
            wrapped(dataclasses.replace(g))
        torch.cuda.synchronize(device)
        wrapped = GraphedEvaluation(wrapped)

    def step():
        out = wrapped(dataclasses.replace(g))
        return out["energy"].sum(), out["forces"], out["stress"]
    return step, g


def compare(a, b):
    """Largest absolute and relative differences of two (energy, forces, stress) results."""
    rows = {}
    for name, x, y in zip(("energy", "forces", "stress"), a, b):
        x, y = x.detach().double().reshape(-1).cpu(), y.detach().double().reshape(-1).cpu()
        d = (x - y).abs().max().item()
        rows[name] = {"max_abs": d, "rel": d / max(y.abs().max().item(), 1e-30)}
    return rows
