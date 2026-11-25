"""Which attention heads carry the class-shared push: attribution patching on saved w = 4 states, checked by exact patching.

A head's output is its 64 channels of the input to attn.proj. Per head h and seed s, z_h is that output in the conditional
branch and z_h' in the null branch, on the same x_t. Patching z_h -> z_h' removes everything about the head that differs with
the class. To first order its effect on a quantity D (cond - uncond, at a site) is r_h = J_h (z_h - z_h'), J_h = dD/dz_h.
With M = mean over seeds of D (shared push) and E_s = D_s - M (seed-specific part), summed over classes:
  shared_h    sum_s <r_h,s, M> / (S ||M||^2)       = <grad of sum_s <D_s, M>, z - z'> / (S ||M||^2)
  specific_h  sum_s <r_h,s, E_s> / sum_s ||E_s||^2 = <grad of <D_s, E_s>, z - z'> summed over s / sum_s ||E_s||^2
so two backward passes per (seed, step, site) give every head at once. Sites: out (guided channels of the velocity) and b15
(block 15's token output; only heads in blocks 0-15 can reach it). Calibration seeds 0-15 only.

--exact K then patches the top K heads (by shared - specific at site b15, then out) one at a time, exactly, on seeds 0-7, to
check the linear estimate. Writes <out>.csv (attribution) and <out>_exact.csv.
usage: head_circuit.py --run <guidance_capture run with x_t> --weights <file> --out <path without extension> [--exact 16]
"""
import argparse, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "bench")
from common import GUIDED_CHANNELS, LABELS, NULL_CLASS, load_model
from guidance_analysis import write_csv

STEPS = [3, 8, 14]  # s_t = 0.06, 0.16, 0.29
SEEDS = range(16)
SITE_BLOCK = 15
CLASS_CHUNK = 2  # classes per backward pass, to keep activation memory near 1-2 GB


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--exact", type=int, default=16)
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    dev = a.device
    model = load_model("SiT-L/2", a.weights, dev)
    for prm in model.parameters():
        prm.requires_grad_(False)
    n_blocks, heads = len(model.blocks), model.blocks[0].attn.num_heads
    hd = model.blocks[0].attn.head_dim
    traj = [np.load(Path(a.run) / "w4" / f"seed_{s}" / "trajectory.npz") for s in SEEDS]
    s_t = traj[0]["s_t"]
    z, site = {}, {}

    def keep_z(b):
        def hook(_m, inputs):
            z[b] = inputs[0]
            if z[b].requires_grad:
                z[b].retain_grad()
        return hook

    def keep_site(_m, _i, out):
        site["b15"] = out

    def run(x, classes, patch=None):
        """Forward on the given classes (cond rows then null rows). Returns (D_out (k, 3072), D_b15 (k, tokens * width))."""
        k = len(classes)
        y = torch.tensor([LABELS[c] for c in classes] + [NULL_CLASS] * k, device=dev)
        handles = [model.blocks[b].attn.proj.register_forward_pre_hook(keep_z(b)) for b in range(n_blocks)]
        handles.append(model.blocks[SITE_BLOCK].register_forward_hook(keep_site))
        if patch is not None:  # (block, head): cond rows take the null rows' head output
            b, h = patch

            def swap(_m, inputs):
                zz = inputs[0].clone()
                zz[:k, :, h * hd:(h + 1) * hd] = zz[k:, :, h * hd:(h + 1) * hd]
                return (zz,)
            handles.insert(0, model.blocks[b].attn.proj.register_forward_pre_hook(swap))
        try:
            xb = x[classes]
            out = model.forward(torch.cat([xb, xb], 0), torch.full((2 * k,), t_now, device=dev), y)
        finally:
            for h_ in handles:
                h_.remove()
        d_out = (out[:k, :GUIDED_CHANNELS] - out[k:, :GUIDED_CHANNELS]).reshape(k, -1)
        d_b15 = (site["b15"][:k] - site["b15"][k:]).reshape(k, -1)
        return d_out, d_b15

    att = {key: np.zeros((len(STEPS), n_blocks, heads, 2)) for key in ("out", "b15")}  # [..., 0] shared, [..., 1] specific
    norms = {}
    chunks = [list(range(i, min(i + CLASS_CHUNK, len(LABELS)))) for i in range(0, len(LABELS), CLASS_CHUNK)]
    for si, k_step in enumerate(STEPS):
        t_now = float(s_t[k_step])
        xs = [torch.from_numpy(tr["x_t"][k_step]).float().to(dev) for tr in traj]
        with torch.no_grad():  # baseline D for every seed, to form M and E
            base = [[torch.cat(v) for v in zip(*[run(x, c) for c in chunks])] for x in xs]
        for key, idx in (("out", 0), ("b15", 1)):
            D = torch.stack([b[idx] for b in base])  # (seeds, 8, dim)
            M = D.mean(0)
            E = D - M
            norms[(si, key)] = (float(len(SEEDS) * (M ** 2).sum()), float((E ** 2).sum()))
            for s, x in enumerate(xs):
                for c in chunks:
                    xg = x.clone().requires_grad_(True)  # needed so the graph (and z's gradients) gets built
                    d_out, d_b15 = run(xg, c)
                    dsite = d_out if key == "out" else d_b15
                    for j, target in enumerate((M[c], E[s, c])):  # shared, then this seed's specific part
                        for b in z:
                            if z[b].grad is not None:
                                z[b].grad = None
                        (dsite * target).sum().backward(retain_graph=j == 0)
                        for b in range(n_blocks):
                            g, zz = z[b].grad, z[b].detach()
                            if g is None:
                                continue  # blocks after the site do not reach it
                            k = len(c)
                            contrib = (g[:k] * (zz[:k] - zz[k:])).reshape(k, -1, heads, hd).sum((0, 1, 3))  # per head
                            att[key][si, b, :, j] += contrib.float().cpu().numpy()
            print(f"step {k_step} site {key} done", flush=True)
    rows = []
    for key in att:
        for si, k_step in enumerate(STEPS):
            ms, es = norms[(si, key)]
            for b in range(n_blocks):
                for h in range(heads):
                    rows.append(dict(step=k_step, t=round(float(s_t[k_step]), 4), site=key, block=b, head=h,
                                     shared=att[key][si, b, h, 0] / ms, specific=att[key][si, b, h, 1] / es))
    write_csv(Path(a.out + ".csv"), rows)

    # Exact check: patch the top heads one at a time on seeds 0-7, compare with the attribution estimate.
    if a.exact:
        score = {}
        for r in rows:
            if r["site"] == "b15":
                score[(r["block"], r["head"])] = score.get((r["block"], r["head"]), 0) + r["shared"] - r["specific"]
        top = sorted(score, key=score.get, reverse=True)[:a.exact]
        exact = []
        k_step = STEPS[1]
        t_now = float(s_t[k_step])
        xs = [torch.from_numpy(tr["x_t"][k_step]).float().to(dev) for tr in traj[:8]]
        with torch.no_grad():
            base = [[torch.cat(v) for v in zip(*[run(x, c) for c in chunks])] for x in xs]
            for key, idx in (("out", 0), ("b15", 1)):
                D = torch.stack([b[idx] for b in base])
                M = D.mean(0)
                E = D - M
                for bh in top:
                    R = D - torch.stack([torch.cat([run(x, c, bh)[idx] for c in chunks]) for x in xs])
                    est = [r for r in rows if r["site"] == key and r["step"] == k_step and (r["block"], r["head"]) == bh][0]
                    exact.append(dict(site=key, block=bh[0], head=bh[1], step=k_step,
                                      shared=float((R.mean(0) * M).sum() / (M ** 2).sum()),
                                      specific=float((R * E).sum() / (E ** 2).sum()),
                                      est_shared=est["shared"], est_specific=est["specific"]))
                print(f"exact {key} done", flush=True)
        write_csv(Path(a.out + "_exact.csv"), exact)


if __name__ == "__main__":
    main()
