"""How many real subclusters do a condition's samples use? Saved centroids (subclusters.py) applied to any run and seed range.

Per class: assignment shares of the samples to the K real-subcluster centroids (DINOv2 cosine). Reported, averaged over classes:
eff (exp entropy of the shares), top (share in the most-used subcluster), tv (total variation to the real shares), and eff over
on-manifold samples only (precision test against the class's 1,300 training images). Intervals for the difference from w4: 1000
paired half-samples of the seeds (these counts depend on the whole sample set, so a bootstrap's duplicates would bias them),
centered on the full-sample difference and scaled by 1/sqrt(2).
usage: subcluster_occupancy.py <run> <seed_lo> <seed_hi> cond ... [--k 8]
"""
import sys
sys.path.insert(0, "bench")
import numpy as np
from pathlib import Path
from common import LABELS
from guidance_analysis import unit
from real_fidelity import knn_radius

args = [x for x in sys.argv[1:]]
K = int(args[args.index("--k") + 1]) if "--k" in args else 8
args = [x for i, x in enumerate(args) if x != "--k" and (i == 0 or args[i - 1] != "--k")]
run, seeds, conds = Path(args[0]), range(int(args[1]), int(args[2]) + 1), args[3:]
conds = ["w4"] + [c for c in conds if c != "w4"]
tab = np.load(f"outputs/guidance_delta/analysis/subclusters_k{K}.npz")
train = np.load("outputs/real_imagenet_train/dino.npz")
R = {l: train["dino"][train["labels"] == l].astype(np.float64) for l in LABELS}; RAD = {l: knn_radius(R[l]) for l in LABELS}
SUB, INS = {}, {}
for c in conds:
    D = np.stack([np.load(run / "embeddings" / c / f"seed_{s}.npz")["dino"] for s in seeds]).astype(np.float64)
    SUB[c] = np.stack([np.argmax(unit(D[:, i]) @ tab[f"centroids_{l}"].T, 1) for i, l in enumerate(LABELS)], 1)
    INS[c] = np.stack([(np.linalg.norm(D[:, i, None] - R[l][None], axis=-1) <= RAD[l][None]).any(1) for i, l in enumerate(LABELS)], 1)


def stats(c, idx):
    eff, top, tv, effi = [], [], [], []
    for i, l in enumerate(LABELS):
        q = np.bincount(SUB[c][idx, i], minlength=K) / len(idx)
        H = lambda q: float(np.exp(-(q[q > 0] * np.log(q[q > 0])).sum()))
        eff.append(H(q)); top.append(q.max()); tv.append(0.5 * np.abs(q - tab[f"share_{l}"]).sum())
        s = SUB[c][idx, i][INS[c][idx, i]]
        effi.append(H(np.bincount(s, minlength=K) / len(s)) if len(s) else 0.0)
    return np.array([np.mean(eff), np.mean(top), np.mean(tv), np.mean(effi)])


rng = np.random.default_rng(0); n = len(seeds)
boots = [np.sort(rng.permutation(n)[: n // 2]) for _ in range(1000)]
B = {c: np.array([stats(c, b) for b in boots]) for c in conds}
print(f"{run.name}, seeds {seeds.start}-{seeds.stop - 1}, K = {K}")
print(f"{'':22} {'eff. subclusters':>24} {'top share':>22} {'TV to real':>12} {'eff. on-manifold':>24}")
for c in conds:
    f = stats(c, np.arange(n))
    f4 = stats("w4", np.arange(n))
    def d(j):
        if c == "w4":
            return ""
        h = B[c][:, j] - B["w4"][:, j]; v = (h - h.mean()) / np.sqrt(2) + f[j] - f4[j]
        return f" ({f[j] - f4[j]:+.2f} [{np.percentile(v, 2.5):+.2f},{np.percentile(v, 97.5):+.2f}])"
    print(f"{c:22} {f[0]:5.2f}{d(0):>19} {f[1]:5.2f}{d(1):>17} {f[2]:12.2f} {f[3]:5.2f}{d(3):>19}")
