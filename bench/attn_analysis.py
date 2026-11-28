"""Analyse attn_capture.py: how guidance changes SiT's attention, and whether seeds come to attend to the same places.

Per block group (0-7, 8-15, 16-23) and phase (window s_t < 0.3, after), mean over heads, classes and seeds:
  class_l1     how much the class changes each head's attention at the same x_t (cond vs uncond rows)
  entropy, distance   attention entropy and mean attention distance of the conditional rows; w4 - w1 paired by seed
  seed_sim     cross-seed similarity of each head's received-attention map: mean cosine over seed pairs of a class of
               (received - 1/256), i.e. of where the head looks beyond uniform. Higher at w = 4 = seeds route attention
               to the same positions (spatial collapse inside the network)
Intervals: 95% over 1000 seed resamples (paired across w; seed_sim over the distinct seed pairs of each resample).
usage: attn_analysis.py <attn run>   -> <run>/attn_summary.csv and attn_blocks.png
"""
import sys
sys.path.insert(0, "bench")
import numpy as np
from pathlib import Path
from guidance_analysis import write_csv

run = Path(sys.argv[1])
WS = ("w1", "w4")
seeds = sorted(set.intersection(*[{int(p.stem.split("_")[1]) for p in (run / w).glob("seed_*.npz")} for w in WS]))
data = {w: [np.load(run / w / f"seed_{s}.npz") for s in seeds] for w in WS}
L = {w: {k: np.stack([d[k] for d in data[w]]) for k in ("class_l1", "entropy", "distance")} for w in WS}  # (S, 49, 24, 8, H)
rsteps = data["w1"][0]["received_steps"]
s_t = np.arange(49) / 49
GROUPS = {"0-7": range(0, 8), "8-15": range(8, 16), "16-23": range(16, 24)}
PHASES = {"window": s_t < 0.3, "after": s_t >= 0.3}


def seed_sim(w):
    """(len(rsteps), 24, S, S): mean over heads and classes of the cosine between seeds' (received - 1/256) maps."""
    out = np.zeros((len(rsteps), 24, len(seeds), len(seeds)))
    for j in range(len(rsteps)):
        R = np.stack([d["received"][j].astype(np.float32) for d in data[w]]) - 1 / 256  # (S, 24, 8, H, 256)
        U = R / (np.linalg.norm(R, axis=-1, keepdims=True) + 1e-12)
        out[j] = np.einsum("sbchn,tbchn->bchst", U, U).mean((1, 2))  # (24, S, S)
    return out


def pair_mean(M, idx):
    """Mean over distinct seed pairs among the (unique) seeds idx of M (..., S, S)."""
    u = np.unique(idx)
    i, j = np.triu_indices(len(u), 1)
    return M[..., u[i], u[j]].mean(-1)


GS = {w: seed_sim(w) for w in WS}
SIM = {w: pair_mean(GS[w], np.arange(len(seeds))) for w in WS}  # (len(rsteps), 24)
rng = np.random.default_rng(0); n = len(seeds)
boots = [np.arange(n)] + [np.unique(rng.choice(n, n)) for _ in range(1000)]
rows = []
for g, blocks in GROUPS.items():
    for ph, mask in PHASES.items():
        row = dict(blocks=g, phase=ph)
        for key in ("class_l1", "entropy", "distance"):
            for w in WS:
                row[f"{key}_{w}"] = float(L[w][key][:, mask][:, :, list(blocks)].mean())
            diff = [float((L["w4"][key][b][:, mask][:, :, list(blocks)] - L["w1"][key][b][:, mask][:, :, list(blocks)]).mean()) for b in boots]
            row[f"{key}_diff"], row[f"{key}_diff_low"], row[f"{key}_diff_high"] = diff[0], np.percentile(diff[1:], 2.5), np.percentile(diff[1:], 97.5)
        rmask = mask[rsteps]
        for w in WS:
            row[f"seed_sim_{w}"] = float(SIM[w][rmask][:, list(blocks)].mean())
        # paired seed resamples (at least 3 unique seeds, so there are pairs)
        sd = [float((pair_mean(GS["w4"], b) - pair_mean(GS["w1"], b))[rmask][:, list(blocks)].mean()) for b in boots if len(np.unique(b)) >= 3]
        row["seed_sim_diff"], row["seed_sim_diff_low"], row["seed_sim_diff_high"] = sd[0], np.percentile(sd[1:], 2.5), np.percentile(sd[1:], 97.5)
        rows.append(row)
        print(f"blocks {g:5} {ph:6}  class_l1 w1 {row['class_l1_w1']:.3f} w4 {row['class_l1_w4']:.3f} | entropy w4-w1 {row['entropy_diff']:+.4f} "
              f"[{row['entropy_diff_low']:+.4f},{row['entropy_diff_high']:+.4f}] | distance w4-w1 {row['distance_diff']:+.3f} "
              f"[{row['distance_diff_low']:+.3f},{row['distance_diff_high']:+.3f}] | seed_sim w1 {row['seed_sim_w1']:.3f} w4 {row['seed_sim_w4']:.3f} (w4-w1 {row['seed_sim_diff']:+.3f} [{row['seed_sim_diff_low']:+.3f},{row['seed_sim_diff_high']:+.3f}])", flush=True)
write_csv(run / "attn_summary.csv", rows)
# Heads whose attention the class changes most, in the window (mean over seeds, classes, window steps), at w = 4.
cl = L["w4"]["class_l1"][:, PHASES["window"]].mean((0, 1, 3))  # (24, H)
top = np.dstack(np.unravel_index(np.argsort(-cl, axis=None)[:10], cl.shape))[0]
print("most class-dependent heads in the window (block.head: class_l1):", ", ".join(f"{b}.{h}: {cl[b, h]:.3f}" for b, h in top))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for i, (key, title) in enumerate((("class_l1", "class effect on attention (cond vs uncond L1)"), ("seed_sim", "cross-seed similarity of attended positions"), ("distance", "mean attention distance (patches)"))):
    for w, col in (("w1", "#2a6fdb"), ("w4", "#d9480f")):
        if key == "seed_sim":
            im = SIM[w]; t = rsteps / 49
        else:
            im = L[w][key].mean((0, 3, 4)); t = s_t
        for g, ls in zip(GROUPS, ("-", "--", ":")):
            ax[i].plot(t, im[:, list(GROUPS[g])].mean(1), ls, color=col, label=f"{w}, blocks {g}")
    ax[i].axvline(0.3, color="#bbb", lw=0.8); ax[i].set(xlabel="s_t", title=title)
ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(run / "attn_blocks.png", dpi=150)
