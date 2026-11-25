"""Within-class statistics of a guidance_capture.py run at w = 4 and w = 1. Writes CSV tables to <run>/analysis/.

Stages and the report figure each feeds:
  stats     alignment_shared.csv   Figure 1: do seeds of one class share Delta, f_cond and f_uncond? (mean pairwise cosine, S)
  subspace  subspace_pooled.csv    Figure 2: effective rank of each block's pooled conditional features, w = 1 vs w = 4

Every cross-seed quantity is computed over the 16 seeds of one class, then averaged over the 8 classes.
Shapes: T = 49 steps, 16 seeds, 8 classes, D = flattened feature size (4096 for a latent, 1024 for a pooled activation).
usage: guidance_analysis.py --run <dir> [--stages stats subspace]

This module also holds helpers the other analysis scripts import: write_csv, unit, pairwise_cos_distance, svd_basis,
effective_rank, mean_pairwise_cos and shared_proportion.
"""
import argparse, csv
from pathlib import Path

import numpy as np

WS = (4, 1)  # the two guidance scales compared
SEEDS = list(range(16))


def write_csv(path, rows):
    """Write a list of dicts as a CSV; the first row's keys become the header. Shared by every analysis script."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_trajectories(run, w, keys):
    """Returns {key: array} with axes (T, seeds, classes, ...)."""
    files = [np.load(run / f"w{w}" / f"seed_{s}" / "trajectory.npz") for s in SEEDS]
    # Each file stores (T, classes, ...); stacking on axis 1 inserts the seed axis after time.
    return {k: np.stack([f[k] for f in files], 1) for k in keys}


def unit(x):
    """Scale every vector along the last axis to length 1."""
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def mean_pairwise_cos(v):
    """v: (N, D) in float64. Mean cosine over distinct pairs, using ||sum of units||^2 = N + sum over ordered pairs."""
    # With unit vectors u_i: ||sum u_i||^2 = sum_i sum_j u_i.u_j = N (the i = j terms, each 1) + sum over the N(N-1) ordered
    # pairs i != j. Solving for that sum and dividing by N(N-1) gives the mean cosine in O(N D) instead of O(N^2 D).
    n = len(v)
    total = unit(v).sum(0)
    return float((total @ total - n) / (n * (n - 1)))


def shared_proportion(v):
    """S = ||mean_i v_i||^2 / mean_i ||v_i||^2. Equals 1 when all v_i match, and about 1/N for independent zero-mean v_i."""
    # Unlike mean_pairwise_cos, S weights each vector by its length: a few long vectors that agree raise S more than many short ones.
    return float(np.sum(v.mean(0) ** 2) / np.mean(np.sum(v ** 2, axis=1)))


def pairwise_cos_distance(x):
    """x: (N, D). Mean cosine distance over distinct pairs: the within-class diversity measure (1 - MSS in the report)."""
    i, j = np.triu_indices(len(x), 1)  # every pair i < j exactly once
    u = unit(x)
    return float((1 - (u[i] * u[j]).sum(1)).mean())  # cosine distance = 1 - cosine similarity


def vendi(x):
    """x: (N, D). Vendi score (Friedman & Dieng 2023) with cosine similarity: exp of the entropy of the eigenvalues of K / N,
    K_ij = cos(x_i, x_j). The effective number of distinct samples: 1 if all are identical, N if all are orthogonal. Unlike the
    mean pairwise distance, it rewards spreading over many directions rather than a few far-apart pairs."""
    u = unit(x)
    mu = np.clip(np.linalg.eigvalsh(u @ u.T / len(x)), 0, None)  # K / N is PSD with trace 1; clip rounding below zero
    mu = mu[mu > 0]
    return float(np.exp(-(mu * np.log(mu)).sum()))


def svd_basis(x):
    """x: (rows, D) already centered. Returns singular values and right singular vectors, using the small Gram matrix."""
    # Here rows (16) is far smaller than D (1024), so a full SVD of x would be wasteful. If x = U S V^T, then x x^T = U S^2 U^T:
    # an eigen-decomposition of the small rows x rows matrix gives U and S, and the right singular vectors follow as
    # V^T = S^-1 U^T x.
    x = x.astype(np.float64)
    vals, vecs = np.linalg.eigh(x @ x.T)
    order = np.argsort(vals)[::-1]  # eigh returns ascending order; sort largest first
    vals, vecs = np.clip(vals[order], 0, None), vecs[:, order]  # clip rounding-level negative eigenvalues to 0
    # Centering removes one dimension, so at least one eigenvalue is 0 up to rounding. Dividing by its tiny sigma would
    # produce garbage directions, so such eigenvalues are dropped.
    keep = vals > vals[0] * 1e-12
    sigma = np.sqrt(vals[keep])
    return sigma, (vecs[:, keep] / sigma).T @ x


def effective_rank(sigma):
    """exp(entropy) of the variance spectrum: how many directions the variation is really spread over.

    p_i = sigma_i^2 / sum sigma^2 is each direction's share of the variance. Equal shares over r directions give exactly r; all
    variance in one direction gives 1. With 16 centered seeds the maximum is 15.
    """
    if len(sigma) == 0:
        return 0.0
    p = sigma ** 2 / np.sum(sigma ** 2)
    return float(np.exp(-np.sum(p * np.log(p))))


def stats_stage(run, out):
    """Per step and w: how much the 16 seeds of a class agree on Delta, f_cond and f_uncond (Figure 1)."""
    alignment = []
    for w in WS:
        d = load_trajectories(run, w, ("s_t", "delta", "f_cond", "f_uncond"))
        t = d["s_t"][:, 0]  # s_t is identical for every seed; take seed 0's copy
        T = len(t)
        # Flatten each latent to one vector: (T, 16 seeds, 8 classes, 4096).
        flat = {k: d[k].reshape(T, 16, 8, -1).astype(np.float64) for k in ("delta", "f_cond", "f_uncond")}
        for k in range(T):
            row = dict(w=w, step=k, t=t[k])
            # Mean pairwise cosine of Delta within each class: the class mean, plus the range over the 8 classes for the band.
            vals = [mean_pairwise_cos(flat["delta"][k][:, c]) for c in range(8)]
            row["pairwise_cos_delta_class"] = np.mean(vals)
            row["pairwise_cos_delta_class_min"], row["pairwise_cos_delta_class_max"] = min(vals), max(vals)
            # Shared proportion of Delta and of each branch's velocity, per class, with the range over classes.
            for name in ("delta", "f_cond", "f_uncond"):
                per_class = [shared_proportion(flat[name][k][:, c]) for c in range(8)]
                row[f"S_{name}_class"] = np.mean(per_class)
                row[f"S_{name}_class_min"], row[f"S_{name}_class_max"] = min(per_class), max(per_class)
            alignment.append(row)
    write_csv(out / "alignment_shared.csv", alignment)


def subspace_stage(run, out):
    """Effective rank of the 16 seeds' pooled conditional features, per block and step, at w = 1 and w = 4 (Figure 2)."""
    # mmap_mode="r" maps the files instead of reading them, so only the slices actually used are loaded from disk.
    pooled = {w: [np.load(run / f"w{w}" / f"seed_{s}" / "pooled.npy", mmap_mode="r") for s in SEEDS] for w in WS}
    t = np.load(run / "w4" / "seed_0" / "trajectory.npz")["s_t"]
    rows = []
    for layer in range(24):
        # Branch 0 = conditional. (16 seeds, T, 8 classes, 1024)
        series = {w: np.stack([np.asarray(p[:, layer, 0]) for p in pooled[w]], 0) for w in WS}
        for k in range(len(t)):
            row = dict(kind="pooled", branch="cond", layer=layer, step=k, t=t[k])
            for w in WS:
                # Center each class's 16 seeds, take the singular values, and average the effective rank over the 8 classes.
                x = series[w][:, k]
                row[f"eff_rank_w{w}"] = np.mean([effective_rank(svd_basis(x[:, c] - x[:, c].mean(0))[0]) for c in range(8)])
            rows.append(row)
    write_csv(out / "subspace_pooled.csv", rows)


def main():
    """Run the requested stages in order; each writes its own CSV into <run>/analysis/."""
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--stages", nargs="+", default=["stats", "subspace"])
    a = p.parse_args()
    run = Path(a.run)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    stages = dict(stats=stats_stage, subspace=subspace_stage)
    for name in a.stages:
        stages[name](run, out)
        print(f"{name} done", flush=True)


if __name__ == "__main__":
    main()
