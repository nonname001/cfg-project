"""What class conditioning writes into each block's tokens, from a guidance_capture.py --token-steps run (w = 4).

For each block, token step and class, with d = cond - uncond block output and u = the unconditional output, both (16 seeds,
256 tokens, 1024 channels), writes <run>/analysis/cond_tokens.csv with:

  push_ratio      ||d|| / ||u||: how large the conditioning push is relative to the activations it is added to
  shared          ||mean_s d||^2 / mean_s ||d_s||^2: share of the push that is the same for every seed (chance level 1/16)
  uniform         share of d's energy in its token average, the part a pooled (per-channel) edit can reach. Feature damping
                  (guidance_intervene.py feat) only edits this part
  top{k}          share of d's energy in its top-k channel directions (fit and measured on all seeds): how low-rank the push is
  heldout{k}      the same with directions fit on seeds 0-7 (this class) and measured on seeds 8-15: what a calibrated
                  subspace intervention could reach on new seeds
  heldout{k}_any  as heldout{k} with one basis fit on all 8 classes together
  overlap16       share of u's across-seed variation (u centered over seeds at each token) lying in d's top-16 directions;
                  chance level is 16/1024. High means conditioning pushes along the directions in which seeds differ
  overlap16_max   the same share for u's own top-16 variation directions, the most any 16 directions could capture
  spatial_top25   share of the shared push's energy in its 25% strongest tokens (1/4 if spread evenly over the image)
  spatial_agree   mean correlation across seed pairs of the per-token push magnitude maps ||d_s(token)||

usage: cond_tokens.py --run <dir> [--blocks 8 9 ...]     (from the project root)
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "bench")
from guidance_analysis import write_csv

KS = (1, 4, 16)
FIT, TEST = slice(0, 8), slice(8, 16)  # seed halves for the held-out measures


def gram(x):
    """x: (..., channels) -> the channels x channels Gram matrix of its rows, in float64."""
    x = x.reshape(-1, x.shape[-1])
    return (x.T @ x).astype(np.float64)


def top_dirs(g, k):
    """g: a Gram matrix. Its top-k eigenvectors (the top right singular vectors of the data) as orthonormal rows."""
    return np.linalg.eigh(g)[1][:, ::-1][:, :k].T  # eigh sorts ascending


def captured(g, basis):
    """Share of the energy of the data behind Gram matrix g inside the span of basis rows: trace(B g B^T) / trace(g)."""
    return float(np.einsum("kc,cd,kd->", basis, g, basis) / np.trace(g))


def class_metrics(d, u):
    """d, u: (seeds, tokens, channels) float32 for one block, step and class. Returns the row and d's fit/test Gram matrices."""
    S, T, C = d.shape
    mean = d.mean(0)
    g_all, g_fit, g_test = gram(d), gram(d[FIT]), gram(d[TEST])
    row = dict(push_ratio=float(np.sqrt(np.trace(g_all) / (u.astype(np.float64) ** 2).sum())),
               shared=float((mean.astype(np.float64) ** 2).sum() / (np.trace(g_all) / S)),
               uniform=float(T * (d.mean(1).astype(np.float64) ** 2).sum() / np.trace(g_all)))
    basis, fit_basis = top_dirs(g_all, max(KS)), top_dirs(g_fit, max(KS))
    for k in KS:
        row[f"top{k}"] = captured(g_all, basis[:k])
        row[f"heldout{k}"] = captured(g_test, fit_basis[:k])
    g_var = gram(u - u.mean(0))  # across-seed variation at each token position
    row["overlap16"] = captured(g_var, basis[:16])
    row["overlap16_max"] = captured(g_var, top_dirs(g_var, 16))
    energy = (mean.astype(np.float64) ** 2).sum(1)  # (tokens,)
    row["spatial_top25"] = float(np.sort(energy)[::-1][: T // 4].sum() / energy.sum())
    corr = np.corrcoef(np.sqrt((d.astype(np.float64) ** 2).sum(2)))  # per-token push magnitude maps, (seeds, tokens)
    row["spatial_agree"] = float(corr[np.triu_indices(S, 1)].mean())
    return row, g_fit, g_test


def analyze(run, blocks):
    folder = run / "w4"
    seeds = sorted(int(p.parent.name[5:]) for p in folder.glob("seed_*/metadata.json"))
    assert len(seeds) == 16, f"expected 16 seeds, found {len(seeds)}"
    steps = json.loads((run / "manifest.json").read_text())["token_steps"]
    s_t = np.load(folder / f"seed_{seeds[0]}" / "trajectory.npz")["s_t"]
    U = [np.load(folder / f"seed_{s}" / "tokens_uncond.npy", mmap_mode="r") for s in seeds]
    D = [np.load(folder / f"seed_{s}" / "tokens_delta.npy", mmap_mode="r") for s in seeds]
    n_classes = D[0].shape[2]
    rows = []
    for i, step in enumerate(steps):
        for b in blocks:
            # One class at a time keeps memory to a few hundred MB: (16 seeds, 256, 1024) float32 per array.
            block_rows, fits, tests = [], [], []
            for c in range(n_classes):
                d = np.stack([x[i, b, c] for x in D], 0).astype(np.float32)
                u = np.stack([x[i, b, c] for x in U], 0).astype(np.float32)
                row, g_fit, g_test = class_metrics(d, u)
                block_rows.append(dict(step=step, t=round(float(s_t[step]), 4), block=b, cls=c, **row))
                fits.append(g_fit)
                tests.append(g_test)
            # One basis for all classes (the Gram of stacked data is the sum of the Grams): does a class-agnostic subspace
            # carry the push as well as per-class ones?
            any_basis = top_dirs(sum(fits), max(KS))
            for row, g_test in zip(block_rows, tests):
                row.update({f"heldout{k}_any": captured(g_test, any_basis[:k]) for k in KS})
            rows += block_rows
            print(f"step {step} block {b} done", flush=True)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    write_csv(out / "cond_tokens.csv", rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--blocks", type=int, nargs="+", default=list(range(24)))
    a = p.parse_args()
    analyze(Path(a.run), a.blocks)


if __name__ == "__main__":
    main()
