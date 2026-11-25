"""Point values of precision retention and DINO / CLIP recovery (as in paired_vs_cond.py) for conditions on a seed range.
usage: points.py <run> <seed_lo> <seed_hi> cond ...     (from the project root)
"""
import sys; sys.path.insert(0, 'bench')
import numpy as np
from pathlib import Path
from common import LABELS
from real_fidelity import knn_radius
from guidance_analysis import pairwise_cos_distance
run = Path(sys.argv[1]); seeds = range(int(sys.argv[2]), int(sys.argv[3]) + 1)
real = np.load('outputs/real_imagenet_val/dino.npz'); R = {c: real['dino'][real['labels'] == c].astype(np.float64) for c in LABELS}
RAD = {l: knn_radius(R[l]) for l in LABELS}
def stats(c):
    d = np.stack([np.load(run / 'embeddings' / c / f'seed_{s}.npz')['dino'] for s in seeds]).astype(np.float64)
    cl = np.stack([np.load(run / 'embeddings' / c / f'seed_{s}.npz')['clip'] for s in seeds]).astype(np.float64)
    p = np.mean([(np.linalg.norm(d[:, i][:, None] - R[l][None], axis=-1) <= RAD[l][None]).any(1).mean() for i, l in enumerate(LABELS)])
    return p, np.mean([pairwise_cos_distance(d[:, i]) for i in range(8)]), np.mean([pairwise_cos_distance(cl[:, i]) for i in range(8)])
w1, w4 = stats('w1'), stats('w4')
for c in sys.argv[4:]:
    p, d, cl = stats(c)
    print(f"{c:42} precRet {(p - w1[0]) / (w4[0] - w1[0]):.2f} dino {(d - w4[1]) / (w1[1] - w4[1]):.2f} clip {(cl - w4[2]) / (w1[2] - w4[2]):.2f}")
