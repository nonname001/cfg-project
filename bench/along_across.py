"""Split each intervention's change in the class signal into a part along w = 4's signal and a part across it (open direction 4).

Class signal of block b at step k: c = pooled conditional - pooled unconditional activation (RESULTS.md 11). With A the
intervened run's signal and C the w = 4 run's, at the same seed, class, block and step:
  along   <A - C, C^> / |C|          signed: < 0 weaker than w = 4 along its own direction, > 0 stronger
  across  |A - C - <A - C, C^> C^| / |C|   redirection: change perpendicular to w = 4's signal
Both are fractions of |C|, so along^2 + across^2 = (|A - C| / |C|)^2, the unsigned total of section 11.1. In the window the
total splits further into direct (A - B) and through x_t (B - C), with B the plain network's signal on the intervened run's
x_t (from --recompute); each part is split along/across in the same way. After the window there is no direct part (A = B).
The class signal is also split into its shared part (mean over seeds of a class) and seed-specific part (the rest), and
"along" is reported for each, against w = 4's shared and seed-specific parts.
usage: along_across.py --run <mech run> --recompute --weights <file>   (GPU; B for steps in the window, cond conditions)
       along_across.py --run <mech run>                                 -> <run>/analysis/along_across.csv
"""
import argparse, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "bench")
from common import LABELS, NULL_CLASS
from guidance_analysis import write_csv

EARLY_END = 0.3
CONDS = ["w1", "score_1", "cond_0.25", "cond_0.5", "cond_1", "score_1+cond_0.5"]
DIRECT = ["cond_0.25", "cond_0.5", "cond_1", "score_1+cond_0.5"]  # conditions that change the network itself in the window


def seeds_of(folder):
    return sorted(int(p.parent.name[5:]) for p in folder.glob("seed_*/metadata.json"))


def recompute(run, weights, device="mps"):
    """B: pooled block outputs of the plain network (no hooks) on each intervened run's own x_t, for steps in the window."""
    import torch
    from common import load_model
    torch.set_grad_enabled(False)
    model = load_model("SiT-L/2", weights, device)
    n = len(LABELS)
    y = torch.tensor(LABELS + [NULL_CLASS] * n, device=device)
    pooled = [None] * len(model.blocks)
    hooks = [blk.register_forward_hook(lambda m, i, o, b=b: pooled.__setitem__(b, o.mean(1))) for b, blk in enumerate(model.blocks)]
    for c in DIRECT:
        for s in seeds_of(run / c):
            target = run / "analysis" / "recomputed" / c / f"seed_{s}.npy"
            if target.exists():
                continue
            traj = np.load(run / c / f"seed_{s}" / "trajectory.npz")
            steps = np.flatnonzero(traj["s_t"] < EARLY_END)
            out = np.empty((len(steps), len(model.blocks), 2, n, pooled_width(model)), np.float32)
            for j, k in enumerate(steps):
                x = torch.from_numpy(np.concatenate([traj["x_t"][k]] * 2)).to(device)
                model.forward(x, torch.full((2 * n,), float(traj["s_t"][k]), device=device), y)
                out[j] = torch.stack(pooled).float().cpu().numpy().reshape(len(model.blocks), 2, n, -1)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target.with_suffix(".tmp.npy"), out); target.with_suffix(".tmp.npy").rename(target)
            print("recomputed", c, s, flush=True)
    for h in hooks:
        h.remove()


def pooled_width(model):
    return model.blocks[0].norm1.normalized_shape[0]


def split(D, C):
    """D, C: (..., 1024). Along (signed) and across parts of D relative to C, as fractions of |C|."""
    nc = np.linalg.norm(C, axis=-1)
    along = (D * C).sum(-1) / nc ** 2
    across = np.linalg.norm(D - along[..., None] * C, axis=-1) / nc
    return along, across


def analyse(run):
    seeds = seeds_of(run / "w4")
    s_t = np.load(run / "w4" / f"seed_{seeds[0]}" / "trajectory.npz")["s_t"]
    window = np.flatnonzero(s_t < EARLY_END)
    mm = {c: [np.load(run / c / f"seed_{s}" / "pooled.npy", mmap_mode="r") for s in seeds] for c in CONDS + ["w4"]}
    # One block at a time from the memory-mapped files: (S, T, 8, 1024) float64 is ~50 MB, the whole run would be ~1.2 GB.
    signal = lambda c, b: np.stack([np.asarray(p[:, b, 0], dtype=np.float64) - p[:, b, 1] for p in mm[c]])
    rec = {c: [np.load(run / "analysis" / "recomputed" / c / f"seed_{s}.npy", mmap_mode="r") for s in seeds] for c in DIRECT}
    rows = []
    for c in CONDS:
        for b in range(24):
            C, A = signal("w4", b), signal(c, b)
            C_shared, A_shared = C.mean(0, keepdims=True), A.mean(0, keepdims=True)
            B = None
            if c in DIRECT:
                B = A.copy()  # after the window B = A; in the window, the recomputed plain-network signal
                B[:, window] = np.stack([np.asarray(r[:, b, 0], dtype=np.float64) - r[:, b, 1] for r in rec[c]])
            elif c == "score_1":
                B = A  # score never changes the network's computation: direct = 0
            parts = dict(total=(A, C))
            if B is not None:
                parts.update(direct=(A, B), through=(B, C))
            for phase, ks in (("window", window), ("after", np.flatnonzero(s_t >= EARLY_END))):
                row = dict(condition=c, phase=phase, block=b)
                for name, (X, Y) in parts.items():
                    al, ac = split((X - Y)[:, ks], C[:, ks])  # every part is measured against w = 4's signal C
                    row[f"{name}_along"], row[f"{name}_across"] = float(al.mean()), float(ac.mean())
                # Shared and seed-specific parts, each against w = 4's own shared / seed-specific part.
                al, _ = split((A_shared - C_shared)[:, ks], C_shared[:, ks])
                row["shared_along"] = float(al.mean())
                al, ac = split(((A - A_shared) - (C - C_shared))[:, ks], (C - C_shared)[:, ks])
                row["specific_along"], row["specific_across"] = float(al.mean()), float(ac.mean())
                row["size_ratio"] = float((np.linalg.norm(A[:, ks], axis=-1) / np.linalg.norm(C[:, ks], axis=-1)).mean())
                rows.append(row)
        for phase in ("window", "after"):
            sub = [r for r in rows if r["condition"] == c and r["phase"] == phase]
            avg = lambda k: np.mean([r[k] for r in sub]) if k in sub[0] else float("nan")
            print(f"{c:18} {phase:6}  total along {avg('total_along'):+.3f} across {avg('total_across'):.3f} | direct along {avg('direct_along'):+.3f} "
                  f"across {avg('direct_across'):.3f} | through along {avg('through_along'):+.3f} across {avg('through_across'):.3f} | "
                  f"shared along {avg('shared_along'):+.3f} | specific along {avg('specific_along'):+.3f} across {avg('specific_across'):.3f} | size {avg('size_ratio'):.3f}", flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))  # w1 rows have no direct / through parts
    write_csv(run / "analysis" / "along_across.csv", [{k: r.get(k, float("nan")) for k in keys} for r in rows])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--weights")
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    if a.recompute:
        recompute(Path(a.run), a.weights, a.device)
    else:
        analyse(Path(a.run))
