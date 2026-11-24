"""Which adaLN modulation components carry the class-shared push: activation patching on saved w = 4 states.

Class conditioning reaches SiT only through each block's adaLN_modulation(c) (six chunks: shift/scale/gate for attention and MLP)
and the final layer's (shift/scale). For a set of such components, this patches the conditional rows' values to the null rows'
(removing their class dependence), reruns one forward pass on a saved state x_t, and measures the removed contribution
r = D - D_patched, where D = cond - uncond, at two sites:
  out    the guided channels of the velocity (the guidance vector Delta)
  b15    block 15's token output, where the token analysis found the class push most shared across seeds (cond_tokens.py)
With M = mean over seeds of D (the shared push) and E_s = D_s - M (the seed-specific part), per class:
  size        sum_s ||r_s||^2 / sum_s ||D_s||^2
  shared      <mean_s r, M> / ||M||^2: share of the shared push the components carry
  specific    sum_s <r_s - mean r, E_s> / sum_s ||E_s||^2: share of the seed-specific part they carry
  ratio       shared / specific: above 1, the components push all seeds the same way more than they shape each seed
Averaged over the 8 classes; one row per (step, configuration, site). Uses calibration seeds 0-15 only.
usage: cond_circuit.py --run <guidance_capture run with x_t> --weights <file> --out <csv>
"""
import argparse, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "bench")
from common import GUIDED_CHANNELS, LABELS, NULL_CLASS, load_model
from guidance_analysis import write_csv

STEPS = [3, 8, 14]  # s_t = 0.06, 0.16, 0.29: inside the early window
SEEDS = range(16)
GROUPS = [range(g, g + 4) for g in range(0, 24, 4)]
CHUNKS = ["shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"]
SITE_BLOCK = 15


def configs():
    """(name, {block: chunk indices}, patch final layer?)."""
    out = [(f"g{g[0]}-{g[-1]}:{CHUNKS[j]}", {b: [j] for b in g}, False) for g in GROUPS for j in range(6)]
    out += [(f"all:{CHUNKS[j]}", {b: [j] for b in range(24)}, False) for j in range(6)]
    out += [(f"g{g[0]}-{g[-1]}:all", {b: list(range(6)) for b in g}, False) for g in GROUPS]
    out += [("final_layer", {}, True), ("everything", {b: list(range(6)) for b in range(24)}, True)]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    torch.set_grad_enabled(False)
    model = load_model("SiT-L/2", a.weights, a.device)
    n = len(LABELS)
    y = torch.tensor(LABELS + [NULL_CLASS] * n, device=a.device)
    traj = [np.load(Path(a.run) / "w4" / f"seed_{s}" / "trajectory.npz") for s in SEEDS]
    s_t = traj[0]["s_t"]
    site = {}

    def site_hook(_m, _i, out):
        site["b15"] = (out[:n] - out[n:]).float()  # (n, tokens, width)

    def patch_hook(chunks, total):
        # total: how many chunks this module's output holds (6 for a block, 2 for the final layer's shift and scale).
        def hook(_m, _i, mod):
            width = mod.shape[1] // total
            cond = mod[:n].clone()
            for j in chunks:
                cond[:, j * width:(j + 1) * width] = mod[n:, j * width:(j + 1) * width]
            return torch.cat([cond, mod[n:]], 0)
        return hook

    def forward(x, t, patch=None, final=False):
        handles = [model.blocks[SITE_BLOCK].register_forward_hook(site_hook)]
        for b, chunks in (patch or {}).items():
            handles.append(model.blocks[b].adaLN_modulation.register_forward_hook(patch_hook(chunks, 6)))
        if final:
            handles.append(model.final_layer.adaLN_modulation.register_forward_hook(patch_hook([0, 1], 2)))
        try:
            out = model.forward(torch.cat([x, x], 0), torch.full((2 * n,), t, device=a.device), y)
        finally:
            for h in handles:
                h.remove()
        delta = (out[:n, :GUIDED_CHANNELS] - out[n:, :GUIDED_CHANNELS]).float().reshape(n, -1)
        return {"out": delta.cpu().numpy().astype(np.float64), "b15": site["b15"].reshape(n, -1).cpu().numpy().astype(np.float64)}

    rows = []
    for k in STEPS:
        xs = [torch.from_numpy(tr["x_t"][k]).float().to(a.device) for tr in traj]
        base = [forward(x, float(s_t[k])) for x in xs]  # per seed: {site: (n, D)}
        for name, patch, final in configs():
            removed = [{key: b[key] - f[key] for key in b} for b, f in zip(base, (forward(x, float(s_t[k]), patch, final) for x in xs))]
            for key in ("out", "b15"):
                D = np.stack([b[key] for b in base])  # (seeds, n, dim)
                R = np.stack([r[key] for r in removed])
                M, Rm = D.mean(0), R.mean(0)  # (n, dim)
                E = D - M
                size = (R ** 2).sum((0, 2)) / (D ** 2).sum((0, 2))
                shared = (Rm * M).sum(1) / (M ** 2).sum(1)
                specific = ((R - Rm) * E).sum((0, 2)) / (E ** 2).sum((0, 2))
                rows.append(dict(step=k, t=round(float(s_t[k]), 4), config=name, site=key, size=float(size.mean()),
                                 shared=float(shared.mean()), specific=float(specific.mean()),
                                 ratio=float(shared.mean() / specific.mean())))
            print(f"step {k} {name} done", flush=True)
    write_csv(Path(a.out), rows)


if __name__ == "__main__":
    main()
