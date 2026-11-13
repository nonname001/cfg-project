"""Scale-free feature measures, and the comparison of w=1, w=4 and the two guidance windows.

Conditions: w1 and w4 from the main run, early = w=4 on [0, 0.3), late = w=4 on [0.3, 1]. All use the same seeds and noise.

Scale-free spread: V(h) = 1 - S(h) = mean_i ||h_i - mean||^2 / mean_i ||h_i||^2, the population variance over the mean squared norm.
The two forms are the same quantity, because mean ||h||^2 = ||mean||^2 + variance. V is unchanged if every h_i is rescaled by one factor.

Reproduced fraction for a metric M: (M_window - M_w1) / (M_w4 - M_w1). 1 means the window recovers the whole w=4 effect, 0 none of it.
Confidence intervals resample seeds, using the same resample in every condition to keep the pairing.

usage: guidance_window.py --base <dir holding the run folders> [--stages scale windows paired]
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "bench")
from guidance_analysis import effective_rank, pairwise_distances, svd_basis, write_csv

SEEDS = list(range(16))
N_BOOT = 1000
LATE_BLOCKS = list(range(13, 24))  # where w=4 lowered effective rank in the main run
LATE_T = 0.3
LOW_CUTOFF, HIGH_CUTOFF = 4, 8  # cycles per 32-pixel latent


def conditions(base):
    main = base / "SiT-L_2_16seeds"
    return {"w1": main / "w1", "w4": main / "w4", "early": base / "SiT-L_2_window_early" / "w4", "late": base / "SiT-L_2_window_late" / "w4"}


def spread_v(x):
    """x: (..., seeds, D) float64 -> V along the seed axis."""
    mean_sq = (x ** 2).sum(-1).mean(-1)
    return ((x - x.mean(-2, keepdims=True)) ** 2).sum(-1).mean(-1) / mean_sq


def scale_stage(base, out):
    """Norms, raw variance and V for pooled and token features, w=4 against w=1, plus the pooled share of token variance."""
    cond = conditions(base)
    manifest = json.loads((base / "SiT-L_2_16seeds" / "manifest.json").read_text())
    t = np.load(cond["w1"] / "seed_0" / "trajectory.npz")["s_t"]
    pooled = {c: np.stack([np.load(cond[c] / f"seed_{s}" / "pooled.npy", mmap_mode="r")[:, :, 0] for s in SEEDS], 0).astype(np.float64)
              for c in ("w1", "w4")}  # (16, T, 24, 8, 1024), conditional branch
    rows = []
    for layer in range(24):
        for k in range(len(t)):
            row = dict(kind="pooled", layer=layer, step=k, t=t[k])
            for c in ("w1", "w4"):
                x = np.moveaxis(pooled[c][:, k, layer], 0, 1)  # (8, 16, 1024)
                row[f"norm_{c}"] = np.linalg.norm(x, axis=-1).mean()
                row[f"var_{c}"] = ((x - x.mean(1, keepdims=True)) ** 2).sum(-1).mean()
                row[f"V_{c}"] = spread_v(x).mean()
            row["raw_ratio"], row["V_ratio"] = row["var_w4"] / row["var_w1"], row["V_w4"] / row["V_w1"]
            row["norm_ratio"] = row["norm_w4"] / row["norm_w1"]
            rows.append(row)
    write_csv(out / "scale_free_pooled.csv", rows)

    tokens = {c: [np.load(cond[c] / f"seed_{s}" / "tokens.npy", mmap_mode="r") for s in SEEDS] for c in ("w1", "w4")}
    rows = []
    for slot, step in enumerate(manifest["token_steps"]):
        for j, layer in enumerate(manifest["token_layers_zero_indexed"]):
            row = dict(kind="tokens", layer=layer, step=step, t=t[step])
            for c in ("w1", "w4"):
                x = np.stack([np.asarray(f[slot, j, 0]) for f in tokens[c]], 1).astype(np.float64)  # (8, 16, 256, 1024)
                # Token variance splits exactly into the part shared by all 256 tokens (the pooled vector) and the spatial pattern:
                # for centered x, sum_tokens ||x||^2 = 256 ||mean over tokens||^2 + sum_tokens ||x - mean over tokens||^2.
                centered = x - x.mean(1, keepdims=True)
                total = (centered ** 2).sum((2, 3)).mean()
                pooled_part = 256 * (centered.mean(2) ** 2).sum(-1).mean()
                row[f"token_norm_{c}"] = np.linalg.norm(x, axis=-1).mean()  # mean per-token norm
                row[f"var_{c}"] = total
                row[f"V_{c}"] = spread_v(x.reshape(8, 16, -1)).mean()
                row[f"pooled_share_{c}"] = pooled_part / total
                row[f"var_spatial_{c}"] = total - pooled_part
            row["raw_ratio"], row["V_ratio"] = row["var_w4"] / row["var_w1"], row["V_w4"] / row["V_w1"]
            row["norm_ratio"] = row["token_norm_w4"] / row["token_norm_w1"]
            row["spatial_ratio"] = row["var_spatial_w4"] / row["var_spatial_w1"]
            rows.append(row)
            print(f"tokens block {layer} step {step}", flush=True)
    write_csv(out / "scale_free_tokens.csv", rows)


def band_masks():
    f = np.fft.fftfreq(32, 1 / 32)
    radius = np.hypot(*np.meshgrid(f, f, indexing="ij"))
    return radius <= LOW_CUTOFF, radius > HIGH_CUTOFF


def band_split(latents):
    """latents: (..., 4, 32, 32). Returns the low-pass and high-pass parts, flattened, as real arrays."""
    low, high = band_masks()
    spectrum = np.fft.fft2(latents, axes=(-2, -1))
    part = lambda mask: np.fft.ifft2(spectrum * mask, axes=(-2, -1)).real.reshape(*latents.shape[:-3], -1)
    return part(low), part(high)


def per_seed_data(cond):
    """Everything the window metrics need, loaded once, with seeds on axis 0."""
    late = None
    data = {}
    for name, path in cond.items():
        files = [np.load(path / f"seed_{s}" / "trajectory.npz") for s in SEEDS]
        t = files[0]["s_t"]
        late = t >= LATE_T
        pooled = np.stack([np.load(path / f"seed_{s}" / "pooled.npy", mmap_mode="r")[:, LATE_BLOCKS, 0][late] for s in SEEDS], 0).astype(np.float32)
        emb = path.parent / "analysis" / "embeddings"
        prefix = "w1" if name == "w1" else "w4"
        data[name] = dict(
            f_cond=np.stack([f["f_cond"][late] for f in files], 0).astype(np.float32), f_uncond=np.stack([f["f_uncond"][late] for f in files], 0).astype(np.float32),
            final=np.stack([f["final"] for f in files], 0).astype(np.float64), pooled=pooled,
            dino=np.stack([np.load(emb / f"{prefix}_seed_{s}.npz")["dino"][-1] for s in SEEDS], 0).astype(np.float64),
            clip=np.stack([np.load(emb / f"{prefix}_seed_{s}.npz")["clip"][-1] for s in SEEDS], 0).astype(np.float64))
    return data


def s_of(x):
    """x: (seeds, T, 8, 4, 32, 32) -> S per class and step, averaged. Duplicate seeds from resampling are removed by the caller."""
    v = x.reshape(*x.shape[:3], -1).astype(np.float64)
    return float((1 - spread_v(np.moveaxis(v, 0, 2))).mean())


def metric_values(d, idx):
    """All window metrics for one condition on the seeds idx (unique, so no duplicate-pair bias)."""
    out = dict(S_f_cond=s_of(d["f_cond"][idx]), S_f_uncond=s_of(d["f_uncond"][idx]))
    x = d["pooled"][idx]  # (seeds, T, blocks, 8, 1024)
    ranks = [effective_rank(svd_basis(x[:, k, b, c] - x[:, k, b, c].mean(0))[0])
             for k in range(0, x.shape[1], 4) for b in range(x.shape[2]) for c in range(8)]
    out["eff_rank_late_blocks"] = float(np.mean(ranks))
    out["V_pooled_late_blocks"] = float(spread_v(np.moveaxis(x.astype(np.float64), 0, 3)).mean())
    for name in ("dino", "clip"):
        out[f"final_{name}"] = float(np.mean([pairwise_distances(d[name][idx, c], "cos") for c in range(8)]))
    final = d["final"][idx]
    out["final_latent_cos"] = float(np.mean([pairwise_distances(final[:, c].reshape(len(idx), -1), "cos") for c in range(8)]))
    low, high = band_split(final)
    for name, band in (("low", low), ("high", high)):
        out[f"V_final_{name}_band"] = float(spread_v(np.moveaxis(band, 0, 1)).mean())
        out[f"final_{name}_band_energy"] = float((band ** 2).sum(-1).mean())
    return out


def curves(base, out):
    """Per-timestep S(f) and late-block effective rank for each condition, for plotting."""
    rows = []
    for name, path in conditions(base).items():
        files = [np.load(path / f"seed_{s}" / "trajectory.npz") for s in SEEDS]
        t = files[0]["s_t"]
        fc = np.stack([f["f_cond"] for f in files], 0)
        fu = np.stack([f["f_uncond"] for f in files], 0)
        pooled = np.stack([np.load(path / f"seed_{s}" / "pooled.npy", mmap_mode="r")[:, LATE_BLOCKS, 0] for s in SEEDS], 0)
        for k in range(len(t)):
            rank = np.mean([effective_rank(svd_basis(pooled[:, k, b, c] - pooled[:, k, b, c].mean(0))[0])
                            for b in range(len(LATE_BLOCKS)) for c in range(8)])
            rows.append(dict(condition=name, step=k, t=t[k], S_f_cond=s_of(fc[:, k:k + 1]), S_f_uncond=s_of(fu[:, k:k + 1]), eff_rank=rank,
                             V_pooled=float(spread_v(np.moveaxis(pooled[:, k].astype(np.float64), 0, 2)).mean())))
    write_csv(out / "window_curves.csv", rows)


def windows_stage(base, out):
    curves(base, out)
    data = per_seed_data(conditions(base))
    names = list(data)
    every = np.arange(16)
    point = {c: metric_values(data[c], every) for c in names}
    rng = np.random.default_rng(0)
    boots = []
    for i in range(N_BOOT // 5):  # the effective-rank metric dominates the run time
        idx = np.unique(rng.choice(16, 16))
        boots.append({c: metric_values(data[c], idx) for c in names})
        if i % 40 == 0:
            print("bootstrap", i, flush=True)
    rows = []
    for metric in point["w1"]:
        row = dict(metric=metric, **{c: point[c][metric] for c in names})
        gap = point["w4"][metric] - point["w1"][metric]
        for c in ("early", "late"):
            row[f"{c}_fraction"] = (point[c][metric] - point["w1"][metric]) / gap
            sample = [(b[c][metric] - b["w1"][metric]) / (b["w4"][metric] - b["w1"][metric]) for b in boots]
            row[f"{c}_ci_low"], row[f"{c}_ci_high"] = np.percentile(sample, [2.5, 97.5])
        rows.append(row)
    write_csv(out / "window_effects.csv", rows)


def paired_stage(base, out):
    """Same-seed comparison of final samples across conditions, with the cross-seed distance as the baseline, plus band spreads.

    If a window leaves a seed's sample close to that seed's sample under another condition, it kept that condition's content.
    """
    cond = conditions(base)
    final = {c: np.stack([np.load(p / f"seed_{s}" / "trajectory.npz")["final"] for s in SEEDS], 0).astype(np.float64) for c, p in cond.items()}
    feats = {}
    for c, p in cond.items():
        prefix = "w1" if c == "w1" else "w4"
        emb = p.parent / "analysis" / "embeddings"
        for name in ("dino", "clip"):
            feats.setdefault(name, {})[c] = np.stack([np.load(emb / f"{prefix}_seed_{s}.npz")[name][-1] for s in SEEDS], 0).astype(np.float64)
    f = np.fft.fftfreq(32, 1 / 32)
    radius = np.hypot(*np.meshgrid(f, f, indexing="ij"))
    masks = dict(low=radius <= LOW_CUTOFF, mid=(radius > LOW_CUTOFF) & (radius <= HIGH_CUTOFF), high=radius > HIGH_CUTOFF)
    for band, mask in masks.items():
        feats[f"latent_{band}"] = {c: np.fft.ifft2(np.fft.fft2(x, axes=(-2, -1)) * mask, axes=(-2, -1)).real.reshape(16, 8, -1) for c, x in final.items()}

    unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    rows = []
    for name, by_cond in feats.items():
        u = {c: unit(x) for c, x in by_cond.items()}
        row = dict(feature=name)
        for a_, b_ in (("late", "w1"), ("late", "w4"), ("early", "w1"), ("early", "w4"), ("w1", "w4")):
            same = 1 - (u[a_] * u[b_]).sum(-1).mean()
            row[f"same_seed_{a_}_{b_}"] = same
            # Baseline: the same two conditions with seeds mismatched, within class.
            row[f"other_seed_{a_}_{b_}"] = np.mean([1 - (u[a_][:, c] @ u[b_][:, c].T)[~np.eye(16, dtype=bool)].mean() for c in range(8)])
        for c in cond:
            row[f"spread_{c}"] = np.mean([pairwise_distances(by_cond[c][:, k], "cos") for k in range(8)])
            row[f"V_{c}"] = spread_v(np.moveaxis(by_cond[c], 0, 1)).mean()
            centroid = unit(u[c].mean(0))
            row[f"to_centroid_{c}"] = 1 - (u[c] * centroid).sum(-1).mean()
        row["energy_w1"], row["energy_w4"] = (by_cond["w1"] ** 2).sum(-1).mean(), (by_cond["w4"] ** 2).sum(-1).mean()
        rows.append(row)
    write_csv(out / "window_paired.csv", rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--stages", nargs="+", default=["scale", "windows", "paired"])
    a = p.parse_args()
    base = Path(a.base)
    out = base / "SiT-L_2_16seeds" / "analysis"
    stages = dict(scale=scale_stage, windows=windows_stage, paired=paired_stage)
    for stage in a.stages:
        stages[stage](base, out)
        print(f"{stage} done", flush=True)


if __name__ == "__main__":
    main()
