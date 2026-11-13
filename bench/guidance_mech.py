"""Mechanistic comparison of score- and condition-space damping, from a guidance_capture.py --conditions run.

Stages:
  curves   per condition and step: |Delta|, |Delta|/|f_uncond|, direction stability, pairwise cos and S of Delta, S of f_cond and f_uncond,
           high-frequency share of Delta, and cross-seed spread of x_hat.
  blocks   per condition, block and step, conditional branch: scale-free spread V, effective rank, norm, and the internal class signal
           pool(h_cond) - pool(h_uncond): its size and how much of it seeds share (S).
  patch    splits each intervention's change to pooled activations and to Delta into a direct part and a trajectory part, by re-running
           the plain model on the intervened run's own x_t:
               total      = logged(intervened x_t, intervention on) - logged w4(w4's x_t)
               direct     = logged(intervened x_t, intervention on) - plain(intervened x_t)
               trajectory = plain(intervened x_t)                   - logged w4(w4's x_t)
           total = direct + trajectory exactly. score only adjusts the model's output, so its internal direct part must be exactly 0.
  finals   copies final latents into the layout guidance_intervene_eval.py reads, for diversity and fidelity.

Cross-seed quantities use the 16 seeds within a class, then average over the 8 classes. Run from the project root.
usage: guidance_mech.py --run <dir> [--stages curves blocks patch finals]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "bench")
from guidance_analysis import effective_rank, fill, mean_pairwise_cos, shared_proportion, svd_basis, write_csv
from guidance_delta import GUIDED_CHANNELS, LABELS, NULL_CLASS, load_model
from guidance_window import spread_v

BASELINES = ("w1", "w4")


def condition_dirs(run):
    return sorted(p for p in run.iterdir() if p.is_dir() and any(p.glob("seed_*/metadata.json")))


def seeds_of(folder):
    return sorted(int(p.parent.name[5:]) for p in folder.glob("seed_*/metadata.json"))


def load(folder, seeds, keys):
    files = [np.load(folder / f"seed_{s}" / "trajectory.npz") for s in seeds]
    return {k: np.stack([f[k] for f in files], 1) for k in keys}  # (T, seeds, classes, ...)


def high_share(delta, cutoff=8):
    f = np.fft.fftfreq(32, 1 / 32)
    mask = np.hypot(*np.meshgrid(f, f, indexing="ij")) <= cutoff
    power = np.abs(np.fft.fft2(delta, axes=(-2, -1), norm="ortho")) ** 2
    total = power.sum((-3, -2, -1))
    low = power[..., mask].sum((-2, -1))  # boolean mask flattens the last two axes, leaving (..., channels, kept bins)
    return float(((total - low) / total).mean())


def curves_stage(run, out):
    rows = []
    for folder in condition_dirs(run):
        seeds = seeds_of(folder)
        d = load(folder, seeds, ("s_t", "delta", "f_cond", "f_uncond", "cos", "x_hat"))
        t, T = d["s_t"][:, 0], len(d["s_t"])
        flat = lambda k: d[k].reshape(T, len(seeds), 8, -1).astype(np.float64)
        delta, fc, fu, xh = flat("delta"), flat("f_cond"), flat("f_uncond"), flat("x_hat")
        for k in range(T):
            row = dict(condition=folder.name, step=k, t=t[k], n_seeds=len(seeds),
                       norm=np.linalg.norm(delta[k], axis=-1).mean(), ratio=(np.linalg.norm(delta[k], axis=-1) / np.linalg.norm(fu[k], axis=-1)).mean(),
                       cos_prev=d["cos"][k - 1].mean() if k else np.nan,
                       pairwise_cos_delta=np.mean([mean_pairwise_cos(delta[k][:, c]) for c in range(8)]),
                       S_delta=np.mean([shared_proportion(delta[k][:, c]) for c in range(8)]),
                       S_f_cond=np.mean([shared_proportion(fc[k][:, c]) for c in range(8)]),
                       S_f_uncond=np.mean([shared_proportion(fu[k][:, c]) for c in range(8)]),
                       high_share=high_share(d["delta"][k].astype(np.float64)),
                       V_x_hat=float(spread_v(np.moveaxis(xh[k], 0, 1)).mean()))
            rows.append(row)
        print("curves", folder.name, flush=True)
    write_csv(out / "mech_curves.csv", rows)


def blocks_stage(run, out):
    rows = []
    for folder in condition_dirs(run):
        seeds = seeds_of(folder)
        pooled = np.stack([np.load(folder / f"seed_{s}" / "pooled.npy", mmap_mode="r") for s in seeds], 0)  # (S, T, 24, 2, 8, 1024)
        t = np.load(folder / f"seed_{seeds[0]}" / "trajectory.npz")["s_t"]
        for b in range(pooled.shape[2]):
            x = pooled[:, :, b].astype(np.float64)  # (S, T, 2, 8, 1024)
            for k in range(len(t)):
                cond, signal = x[:, k, 0], x[:, k, 0] - x[:, k, 1]  # (S, 8, 1024)
                rows.append(dict(condition=folder.name, block=b, step=k, t=t[k],
                                 V=float(spread_v(np.moveaxis(cond, 0, 1)).mean()), norm=float(np.linalg.norm(cond, axis=-1).mean()),
                                 eff_rank=float(np.mean([effective_rank(svd_basis(cond[:, c] - cond[:, c].mean(0))[0]) for c in range(8)])),
                                 signal_norm=float(np.linalg.norm(signal, axis=-1).mean()),
                                 signal_S=float(np.mean([shared_proportion(signal[:, c]) for c in range(8)]))))
        print("blocks", folder.name, flush=True)
    write_csv(out / "mech_blocks.csv", rows)


def patch_stage(run, out, device="mps"):
    manifest = json.loads((run / "manifest.json").read_text())
    torch.set_grad_enabled(False)
    model = load_model(manifest["checkpoint"]["model"], manifest["checkpoint"]["path"], device)
    n = len(LABELS)
    y = torch.tensor(LABELS + [NULL_CLASS] * n, device=device)
    steps = manifest["token_steps"]
    captured = {}
    hooks = [blk.register_forward_hook(lambda _m, _i, o, b=b: captured.__setitem__(b, o.mean(1).float().cpu().numpy())) for b, blk in enumerate(model.blocks)]
    reference = run / "w4"
    rows = []
    try:
        for folder in condition_dirs(run):
            if folder.name in BASELINES:
                continue
            for seed in seeds_of(folder):
                d = np.load(folder / f"seed_{seed}" / "trajectory.npz")
                r = np.load(reference / f"seed_{seed}" / "trajectory.npz")
                pooled = np.load(folder / f"seed_{seed}" / "pooled.npy", mmap_mode="r")
                pooled_ref = np.load(reference / f"seed_{seed}" / "pooled.npy", mmap_mode="r")
                for k in steps:
                    x = torch.from_numpy(d["x_t"][k]).to(device)
                    t = torch.full((2 * n,), float(d["s_t"][k]), device=device)
                    plain_out = model.forward(torch.cat([x, x]), t, y).float().cpu().numpy()
                    plain_delta = plain_out[:n, :GUIDED_CHANNELS] - plain_out[n:, :GUIDED_CHANNELS]
                    for b in range(len(model.blocks)):
                        logged, ref, plain = np.asarray(pooled[k, b], np.float64), np.asarray(pooled_ref[k, b], np.float64), captured[b].reshape(2, n, -1).astype(np.float64)
                        sig = lambda a: a[0] - a[1]  # class signal: conditional minus unconditional
                        scale = np.linalg.norm(sig(ref), axis=-1).mean()
                        rows.append(dict(condition=folder.name, seed=seed, step=k, t=float(d["s_t"][k]), site=f"block{b}", block=b,
                                         direct_cond=np.linalg.norm(logged[0] - plain[0], axis=-1).mean(),
                                         trajectory_cond=np.linalg.norm(plain[0] - ref[0], axis=-1).mean(),
                                         total_cond=np.linalg.norm(logged[0] - ref[0], axis=-1).mean(),
                                         direct_signal=np.linalg.norm(sig(logged) - sig(plain), axis=-1).mean() / scale,
                                         trajectory_signal=np.linalg.norm(sig(plain) - sig(ref), axis=-1).mean() / scale,
                                         total_signal=np.linalg.norm(sig(logged) - sig(ref), axis=-1).mean() / scale))
                    logged_delta = d["delta"][k][:, :GUIDED_CHANNELS].astype(np.float64)
                    ref_delta = r["delta"][k][:, :GUIDED_CHANNELS].astype(np.float64)
                    scale = np.linalg.norm(ref_delta.reshape(n, -1), axis=-1).mean()
                    norm = lambda a: np.linalg.norm(a.reshape(n, -1), axis=-1).mean() / scale
                    rows.append(dict(condition=folder.name, seed=seed, step=k, t=float(d["s_t"][k]), site="delta", block=-1,
                                     direct_signal=norm(logged_delta - plain_delta), trajectory_signal=norm(plain_delta - ref_delta),
                                     total_signal=norm(logged_delta - ref_delta)))
            print("patch", folder.name, flush=True)
    finally:
        for h in hooks:
            h.remove()
    write_csv(out / "mech_patch.csv", [fill(r, rows) for r in rows])  # Delta rows have no activation columns


def finals_stage(run, out):
    """Write <out>/finals/<condition>/seed_<n>.npz with the keys guidance_intervene_eval.py expects."""
    for folder in condition_dirs(run):
        target = out / "finals" / folder.name
        target.mkdir(parents=True, exist_ok=True)
        for seed in seeds_of(folder):
            d = np.load(folder / f"seed_{seed}" / "trajectory.npz")
            meta = json.loads((folder / f"seed_{seed}" / "metadata.json").read_text())
            removed = d["removed"] if "removed" in d else np.zeros(len(d["s_t"]))
            np.savez(target / f"seed_{seed}.npz", seed=seed, z=d["z"], final=d["final"], removed=removed, seconds=meta["seconds"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--stages", nargs="+", default=["curves", "blocks", "patch", "finals"])
    a = p.parse_args()
    run = Path(a.run)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    stages = dict(curves=curves_stage, blocks=blocks_stage, patch=patch_stage, finals=finals_stage)
    for stage in a.stages:
        stages[stage](run, out)
        print(f"{stage} done", flush=True)


if __name__ == "__main__":
    main()
