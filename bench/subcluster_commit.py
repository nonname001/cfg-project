"""When is a sample's subcluster chosen? Subcluster identity across the switch sweep (switch_sweep.py, seeds 0-15).

Each final image is assigned to its nearest real-subcluster centroid (subclusters.py, DINOv2 cosine). For the samples whose
subcluster differs between their w = 1 and w = 4 images:
  w4to1_k   (w = 4 before step k, w = 1 after): share that ends in the w = 4 subcluster, i.e. already chosen by guidance before k
  w1to4_k   (w = 1 before step k, w = 4 after): share that still ends in the w = 4 subcluster, i.e. chosen by guidance after k
Also the share of all samples whose subcluster changes between w = 1 and w = 4, and how concentrated the w = 4 choices are.
Intervals: bootstrap over samples.
usage: subcluster_commit.py [--k 8]
"""
import sys
sys.path.insert(0, "bench")
import numpy as np
from pathlib import Path
from common import LABELS
from guidance_analysis import unit

K = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 8
tab = np.load(f"outputs/guidance_delta/analysis/subclusters_k{K}.npz")
run = Path("outputs/guidance_delta/SiT-L_2_switch"); seeds = range(16)
conds = sorted(p.name for p in (run / "embeddings").iterdir() if p.is_dir())
sub = {}
for c in conds:
    D = np.stack([np.load(run / "embeddings" / c / f"seed_{s}.npz")["dino"] for s in seeds]).astype(np.float64)
    sub[c] = np.stack([np.argmax(unit(D[:, i]) @ tab[f"centroids_{l}"].T, 1) for i, l in enumerate(LABELS)], 1).ravel()
diff = sub["w1"] != sub["w4"]
print(f"seeds 0-15, K = {K}: {diff.mean():.2f} of samples ({diff.sum()}) are in a different subcluster at w = 4 than at w = 1")
rng = np.random.default_rng(0); idx = np.flatnonzero(diff); boots = [rng.choice(idx, len(idx)) for _ in range(1000)]
ks = sorted({int(c.split("_k")[1]) for c in conds if "_k" in c})
print(f"{'switch s_t':>10}  {'w4 before, w1 after: in w4 subcluster':>40}  {'w1 before, w4 after: in w4 subcluster':>40}")
for k in ks:
    a, b = ((sub[f"{d}_k{k}"] == sub["w4"]) for d in ("w4to1", "w1to4"))
    ci = lambda v: f"{v[idx].mean():.2f} [{np.percentile([v[x].mean() for x in boots], 2.5):.2f},{np.percentile([v[x].mean() for x in boots], 97.5):.2f}]"
    print(f"{k / 49:10.2f}  {ci(a):>40}  {ci(b):>40}")
