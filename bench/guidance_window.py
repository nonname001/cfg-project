"""Early versus late guidance (report Table 2): how much of each w = 4 effect guidance in only one window reproduces.

Conditions: w1 and w4 from the main 16-seed run, early = w=4 on [0, 0.3), late = w=4 on [0.3, 1]. All use the same seeds and noise.

Reproduced fraction for a metric M: (M_window - M_w1) / (M_w4 - M_w1). 1 means the window recovers the whole w=4 effect, 0 none of it.
Confidence intervals resample seeds, using the same resample in every condition to keep the pairing.

Metrics (one per row of Table 2):
  eff_rank_late_blocks   effective rank of the seeds' pooled features in blocks 13-23, at t >= 0.3
  final_latent_cos       final-latent diversity: mean pairwise cosine distance across seeds
  final_dino, final_clip final-image diversity in DINOv2 and CLIP embedding space

Also defines spread_v (the scale-free spread V) and band_split (low/high spatial-frequency parts of a latent), which
guidance_intervene_eval.py imports for layout diversity.
usage: guidance_window.py --base <dir holding the run folders>
"""
import argparse, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "bench")
from guidance_analysis import effective_rank, pairwise_cos_distance, svd_basis, write_csv

SEEDS = list(range(16))
N_BOOT = 1000  # nominal; the bootstrap below uses N_BOOT // 5 = 200 resamples because the effective-rank metric is slow
LATE_BLOCKS = list(range(13, 24))  # where w=4 lowered effective rank in the main run
LATE_T = 0.3  # effective rank is measured only after the window boundary, where early and late guidance differ
LOW_CUTOFF, HIGH_CUTOFF = 4, 8  # cycles per 32-pixel latent


def conditions(base):
    """Folder of each of the four conditions. Both window runs store their batches under w4/ because they use w = 4."""
    main = base / "SiT-L_2_16seeds"
    return {"w1": main / "w1", "w4": main / "w4", "early": base / "SiT-L_2_window_early" / "w4", "late": base / "SiT-L_2_window_late" / "w4"}


def spread_v(x):
    """x: (..., seeds, D) float64 -> V along the seed axis.

    V = mean_i ||h_i - mean||^2 / mean_i ||h_i||^2, the variance over the mean squared norm (equal to 1 - S). It is unchanged if
    every h_i is rescaled by one factor, so it measures spread independently of overall scale.
    """
    mean_sq = (x ** 2).sum(-1).mean(-1)  # mean over seeds of ||h_i||^2
    return ((x - x.mean(-2, keepdims=True)) ** 2).sum(-1).mean(-1) / mean_sq


def band_masks():
    """Boolean (32, 32) FFT masks: low = radius <= 4 cycles (coarse layout), high = radius > 8 (fine detail). 4-8 is left out."""
    f = np.fft.fftfreq(32, 1 / 32)  # integer frequencies -16..15, in cycles per 32-pixel image
    radius = np.hypot(*np.meshgrid(f, f, indexing="ij"))
    return radius <= LOW_CUTOFF, radius > HIGH_CUTOFF


def band_split(latents):
    """latents: (..., 4, 32, 32). Returns the low-pass and high-pass parts, flattened, as real arrays."""
    low, high = band_masks()
    spectrum = np.fft.fft2(latents, axes=(-2, -1))
    # Zero every frequency outside the band, transform back, and flatten channels x pixels into one vector per latent.
    part = lambda mask: np.fft.ifft2(spectrum * mask, axes=(-2, -1)).real.reshape(*latents.shape[:-3], -1)
    return part(low), part(high)


def per_seed_data(cond):
    """Everything the window metrics need, loaded once, with seeds on axis 0."""
    data = {}
    for name, path in cond.items():
        t = np.load(path / f"seed_{SEEDS[0]}" / "trajectory.npz")["s_t"]
        late = t >= LATE_T
        # Late blocks, conditional branch, late steps only: (16 seeds, late steps, 11 blocks, 8 classes, 1024).
        pooled = np.stack([np.load(path / f"seed_{s}" / "pooled.npy", mmap_mode="r")[:, LATE_BLOCKS, 0][late] for s in SEEDS], 0).astype(np.float32)
        # Embeddings live in each run's own analysis folder; the window runs name theirs w4_* because they ran at w = 4.
        emb = path.parent / "analysis" / "embeddings"
        prefix = "w1" if name == "w1" else "w4"
        data[name] = dict(
            final=np.stack([np.load(path / f"seed_{s}" / "trajectory.npz")["final"] for s in SEEDS], 0).astype(np.float64), pooled=pooled,
            # [-1] is the last embedding row, the final sample.
            dino=np.stack([np.load(emb / f"{prefix}_seed_{s}.npz")["dino"][-1] for s in SEEDS], 0).astype(np.float64),
            clip=np.stack([np.load(emb / f"{prefix}_seed_{s}.npz")["clip"][-1] for s in SEEDS], 0).astype(np.float64))
    return data


def metric_values(d, idx):
    """The four Table 2 metrics for one condition on the seeds idx (unique, so no duplicate-pair bias)."""
    out = {}
    x = d["pooled"][idx]  # (seeds, T, blocks, 8, 1024)
    # Effective rank of the seeds per (step, block, class). Every 4th step only, because this is the slow part of the bootstrap.
    ranks = [effective_rank(svd_basis(x[:, k, b, c] - x[:, k, b, c].mean(0))[0])
             for k in range(0, x.shape[1], 4) for b in range(x.shape[2]) for c in range(8)]
    out["eff_rank_late_blocks"] = float(np.mean(ranks))
    # Diversity of the final images: mean pairwise cosine distance within each class, in embedding and latent space.
    for name in ("dino", "clip"):
        out[f"final_{name}"] = float(np.mean([pairwise_cos_distance(d[name][idx, c]) for c in range(8)]))
    final = d["final"][idx]
    out["final_latent_cos"] = float(np.mean([pairwise_cos_distance(final[:, c].reshape(len(idx), -1)) for c in range(8)]))
    return out


def windows_stage(base, out):
    """How much of each w=4 effect the early and the late window reproduce, with paired bootstrap intervals."""
    data = per_seed_data(conditions(base))
    names = list(data)
    every = np.arange(16)
    point = {c: metric_values(data[c], every) for c in names}  # point estimates on all 16 seeds
    rng = np.random.default_rng(0)  # fixed seed, so the intervals are reproducible
    boots = []
    for i in range(N_BOOT // 5):  # the effective-rank metric dominates the run time
        # One resample of seeds, used for all four conditions so the comparison stays paired by seed. np.unique keeps each
        # drawn seed once: a duplicated seed would pair with itself at distance 0 and bias diversity down.
        idx = np.unique(rng.choice(16, 16))
        boots.append({c: metric_values(data[c], idx) for c in names})
        if i % 40 == 0:
            print("bootstrap", i, flush=True)
    rows = []
    for metric in point["w1"]:
        row = dict(metric=metric, **{c: point[c][metric] for c in names})
        gap = point["w4"][metric] - point["w1"][metric]  # the full effect of guidance on this metric
        for c in ("early", "late"):
            row[f"{c}_fraction"] = (point[c][metric] - point["w1"][metric]) / gap
            sample = [(b[c][metric] - b["w1"][metric]) / (b["w4"][metric] - b["w1"][metric]) for b in boots]
            row[f"{c}_ci_low"], row[f"{c}_ci_high"] = np.percentile(sample, [2.5, 97.5])
        rows.append(row)
    write_csv(out / "window_effects.csv", rows)


def main():
    """Write window_effects.csv into the main 16-seed run's analysis folder."""
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    a = p.parse_args()
    base = Path(a.base)
    windows_stage(base, base / "SiT-L_2_16seeds" / "analysis")
    print("windows done", flush=True)


if __name__ == "__main__":
    main()
