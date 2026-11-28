"""Intervals for the identity probes (probe_identity.py) at key cells, and a permutation null.

Point values are the full 16-seed probes (checked against probe_identity.csv). Intervals: 300 random subsets of 12 of the 16
seeds, each probed with 4-fold cross-validation over its own seeds (3 held out per fold); the 2.5-97.5 percentiles of R^2 at
w = 1, at w = 4 and of the paired difference (same subset). With 12 seeds instead of 16 the probes are slightly weaker and the
intervals slightly wider than for the full data: conservative. Null: the targets are shuffled across seeds within each class
(100 permutations, full 16 seeds), giving the R^2 a probe reaches by chance.
usage: probe_ci.py <16-seed run>   -> stdout
"""
import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
import csv, sys
sys.path.insert(0, "bench")
import numpy as np
from pathlib import Path
from guidance_analysis import unit
from probe_identity import SEEDS, WS, load_embeddings, r2

run = Path(sys.argv[1])
emb = load_embeddings(run)
TARGET = {w: unit(emb[w][:, -1]) for w in WS}
pooled = {w: [np.load(run / w / f"seed_{s}" / "pooled.npy", mmap_mode="r") for s in SEEDS] for w in WS}
saved = {(r["w"], int(r["step"])): r for r in csv.DictReader(open(run / "analysis" / "probe_identity.csv"))}
BLOCKS, STEPS = [4, 10, 13, 15, 16, 17, 20], [4, 8, 12, 15, 21, 30]
rng = np.random.default_rng(0)
subsets = [np.sort(rng.choice(16, 12, replace=False)) for _ in range(300)]
folds12 = [np.arange(f, 12, 4) for f in range(4)]
print("cell            w1 R2 [95%]            w4 R2 [95%]            w4 - w1 [95%]          null 97.5% (w4)")
for k in STEPS:
    P = {w: np.stack([np.asarray(p[k, BLOCKS, 0], dtype=np.float64) for p in pooled[w]]) for w in WS}  # (16, blocks, 8, 1024)
    for j, b in enumerate(BLOCKS):
        full = {w: r2(P[w][:, j], TARGET[w]) for w in WS}
        assert all(abs(full[w] - float(saved[(w, k)][f"block_{b}"])) < 1e-9 for w in WS), "does not reproduce probe_identity.csv"
        sub = np.array([[r2(P[w][idx, j], TARGET[w][idx], folds=folds12) for w in WS] for idx in subsets])
        diff = sub[:, 1] - sub[:, 0]
        null = []
        if b in (15, 16) and k in (8, 15):
            for _ in range(100):
                perm = np.stack([rng.permutation(16) for _ in range(8)], 1)  # shuffle seeds within each class
                null.append(r2(P["w4"][:, j], TARGET["w4"][perm, np.arange(8)[None]]))
        ci = lambda v: f"[{np.percentile(v, 2.5):+.2f},{np.percentile(v, 97.5):+.2f}]"
        nl = f"{np.percentile(null, 97.5):+.3f}" if null else ""
        print(f"step {k:2d} blk {b:2d}   {full['w1']:+.2f} {ci(sub[:, 0])}     {full['w4']:+.2f} {ci(sub[:, 1])}     "
              f"{full['w4'] - full['w1']:+.2f} {ci(diff)}     {nl}", flush=True)
