"""Fit the per-class shared guidance directions that the `score` method removes.

For each class, average Delta = v_c - v_u (guided channels) over the calibration seeds at every step with t < t_end,
at w = 4 and w = 1 (an intervened trajectory lies between the two). Seed-specific parts cancel in the average; the
top-k right singular vectors of the stacked means span the shared part. Only the early steps are sampled.
usage: python fit_score.py [--seeds 0-15] [--rank 3] [--t-end 0.3]
Calibration seeds must differ from evaluation seeds; methods.Score refuses an overlap.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from common import DEFAULT_WEIGHTS, GUIDED_CHANNELS, load_model, parse_seeds, sample
from methods import CFG


class Recorder(CFG):
    def start(self, model, n, seed):
        super().start(model, n, seed)
        self.deltas = []

    def guide(self, v_c, v_u, x, t):
        self.deltas.append((v_c - v_u)[:, :GUIDED_CHANNELS].flatten(1).cpu().double())  # MPS has no float64
        return super().guide(v_c, v_u, x, t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", default=["0-15"])
    p.add_argument("--rank", type=int, default=3)
    p.add_argument("--t-end", type=float, default=0.3)
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--out", default=str(Path(__file__).parent / "runs" / "score_basis.npz"))
    a = p.parse_args()

    torch.set_grad_enabled(False)
    seeds = parse_seeds(a.seeds)
    model = load_model(a.weights)
    means = []
    for w in (4.0, 1.0):
        per_seed = []
        for seed in seeds:
            rec = Recorder(w)
            sample(model, rec, seed, stop_at=a.t_end)
            per_seed.append(torch.stack(rec.deltas))  # (steps, classes, 3072)
            print(f"w={w} seed {seed}: {len(rec.deltas)} steps", flush=True)
        means.append(torch.stack(per_seed).mean(0))
    stacked = torch.cat(means).numpy()  # (2 * steps, classes, 3072)
    basis, share = [], []
    for c in range(stacked.shape[1]):
        _, sv, vt = np.linalg.svd(stacked[:, c], full_matrices=False)
        basis.append(vt[: a.rank])
        share.append((sv[: a.rank] ** 2).sum() / (sv ** 2).sum())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, basis=np.stack(basis).astype(np.float32), seeds=np.array(seeds), rank=a.rank, t_end=a.t_end)
    print(f"saved {a.out}; variance of the mean Delta captured by rank {a.rank}: "
          + ", ".join(f"{s:.3f}" for s in share))


if __name__ == "__main__":
    main()
