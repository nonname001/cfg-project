"""Evaluate guidance_intervene.py outputs: diversity against class fidelity for every condition, on the same seeds.

Diversity: scale-free low-band latent spread V, DINOv2 and CLIP pairwise cosine distance across seeds within a class.
Fidelity: an ImageNet classifier's probability and top-1 accuracy for the intended class. This is a proxy: FID and precision/recall need
real ImageNet images, which are not available here.

recovery = (M - M_w4) / (M_w1 - M_w4): how much of the diversity lost at w=4 a condition gets back.
retention = (P - P_w1) / (P_w4 - P_w1): how much of the fidelity gained at w=4 it keeps.
Intervals resample seeds, identically across conditions.

usage: guidance_intervene_eval.py --run <dir> [--embed] [--stats] [--figures [--sheet-seed n]]
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "bench")
from guidance_analysis import pairwise_distances, write_csv
from guidance_delta import LABELS
from guidance_window import band_split, spread_v

CLASSIFIER = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
N_BOOT = 1000


def embed(run, device="mps"):
    import timm
    import torch.nn.functional as F
    from diffusers.models import AutoencoderKL
    from torchvision.utils import save_image
    from guidance_embed import CLIP, DINO, LATENT_SCALE, SIZE, VAE, encoder
    torch.set_grad_enabled(False)
    vae = AutoencoderKL.from_pretrained(VAE).to(device).eval()
    dino = encoder(DINO, device, num_classes=0, img_size=SIZE)
    clip = encoder(CLIP, device)
    classifier = encoder(CLASSIFIER, device)
    labels = torch.tensor(LABELS, device=device)
    for path in sorted(run.glob("*/seed_*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        target = run / "embeddings" / path.parent.name / path.name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        z = torch.from_numpy(np.load(path)["final"]).to(device)
        image = ((vae.decode(z / LATENT_SCALE).sample + 1) / 2).clamp(0, 1)
        small = F.interpolate(image, size=SIZE, mode="bicubic", antialias=True).clamp(0, 1)
        run_enc = lambda enc: enc[0]((small - enc[1]) / enc[2]).float()
        logits = run_enc(classifier)
        prob = logits.softmax(-1)
        rows = torch.arange(len(LABELS), device=device)
        (run / "samples").mkdir(exist_ok=True)
        save_image(image.cpu(), run / "samples" / f"{path.parent.name}_{path.stem}.png", nrow=4)
        tmp = target.with_suffix(".tmp.npz")
        np.savez(tmp, dino=run_enc(dino).cpu().numpy(), clip=run_enc(clip).cpu().numpy(), p_target=prob[rows, labels].cpu().numpy(),
                 top1=(logits.argmax(-1) == labels).cpu().numpy())
        tmp.rename(target)
        print("embedded", path.parent.name, path.stem, flush=True)


def load(run):
    data = {}
    for folder in sorted(run.iterdir()):
        if not folder.is_dir() or folder.name in ("embeddings", "samples"):
            continue
        seeds = sorted(int(p.stem.split("_")[1]) for p in folder.glob("seed_*.npz") if not p.name.endswith(".tmp.npz"))
        data[folder.name] = seeds
    shared = sorted(set.intersection(*[set(v) for v in data.values()]))
    out = {}
    for condition in data:
        emb = [np.load(run / "embeddings" / condition / f"seed_{s}.npz") for s in shared]
        out[condition] = dict(final=np.stack([np.load(run / condition / f"seed_{s}.npz")["final"] for s in shared], 0).astype(np.float64),
                              removed=np.stack([np.load(run / condition / f"seed_{s}.npz")["removed"] for s in shared], 0),
                              seconds=np.array([float(np.load(run / condition / f"seed_{s}.npz")["seconds"]) for s in shared]),
                              dino=np.stack([e["dino"] for e in emb], 0).astype(np.float64), clip=np.stack([e["clip"] for e in emb], 0).astype(np.float64),
                              p_target=np.stack([e["p_target"] for e in emb], 0), top1=np.stack([e["top1"] for e in emb], 0))
    return out, shared


def metrics(d, idx):
    low, high = band_split(d["final"][idx])
    return dict(V_low=float(spread_v(np.moveaxis(low, 0, 1)).mean()),
                dino=float(np.mean([pairwise_distances(d["dino"][idx, c], "cos") for c in range(8)])),
                clip=float(np.mean([pairwise_distances(d["clip"][idx, c], "cos") for c in range(8)])),
                p_target=float(d["p_target"][idx].mean()), top1=float(d["top1"][idx].mean()), high_energy=float((high ** 2).sum(-1).mean()))


def stats(run):
    data, seeds = load(run)
    every = np.arange(len(seeds))
    point = {c: metrics(d, every) for c, d in data.items()}
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(N_BOOT):
        idx = np.unique(rng.choice(len(seeds), len(seeds)))
        boots.append({c: metrics(d, idx) for c, d in data.items()})
    rows = []
    for c, m in point.items():
        row = dict(condition=c, n_seeds=len(seeds), mean_seconds=float(np.median(data[c]["seconds"])), removed_step0=float(data[c]["removed"][:, 0].mean()),
                   removed_early=float(data[c]["removed"][:, :15].mean()), **m)
        for key in ("V_low", "dino", "clip"):
            f = lambda b: (b[c][key] - b["w4"][key]) / (b["w1"][key] - b["w4"][key])
            row[f"recovery_{key}"] = f(point)
            row[f"recovery_{key}_low"], row[f"recovery_{key}_high"] = np.percentile([f(b) for b in boots], [2.5, 97.5])
        f = lambda b: (b[c]["p_target"] - b["w1"]["p_target"]) / (b["w4"]["p_target"] - b["w1"]["p_target"])
        row["retention"] = f(point)
        row["retention_low"], row["retention_high"] = np.percentile([f(b) for b in boots], [2.5, 97.5])
        rows.append(row)
    write_csv(run / "intervention_results.csv", rows)
    # Direct paired tests against the scalar control at the same strength.
    pairs = []
    for method in ("score", "freq", "cond", "feat"):
        for strength in ("0.5", "1"):
            a, b = f"{method}_{strength}", f"scalar_{strength}"
            if a not in point or b not in point:
                continue
            for key in ("V_low", "dino", "clip", "p_target"):
                diffs = [x[a][key] - x[b][key] for x in boots]
                low, high = np.percentile(diffs, [2.5, 97.5])
                pairs.append(dict(condition=a, versus=b, metric=key, difference=point[a][key] - point[b][key], ci_low=low, ci_high=high))
    if pairs:  # only when the run includes the scalar control
        write_csv(run / "intervention_vs_scalar.csv", pairs)
    return rows


def figures(run, sheet_seed):
    """Diversity against fidelity for every method, plus a sample sheet for one held-out seed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    from guidance_figures import INK, MUTED, SURFACE, GRID, header, save
    rows = {r["condition"]: r for r in csv.DictReader(open(run / "intervention_results.csv"))}
    v = lambda c, k: float(rows[c][k])
    # Categorical slots 1-5 of the reference palette, in fixed order. Five crossing series would collide as direct labels, so a legend names them.
    methods = [("scalar", "#2a78d6", "scalar w (control)"), ("score", "#eb6834", "score: shared subspace"), ("freq", "#1baf7a", "frequency: low band"),
               ("cond", "#eda100", "condition embedding"), ("feat", "#e87ba4", "features: blocks 13–23")]
    out = run / "figures"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), dpi=170, facecolor=SURFACE)
    for ax, key, label in zip(axes, ("V_low", "dino", "clip"), ("layout diversity: low-band latent V", "DINOv2 diversity", "CLIP diversity")):
        for method, color, name in methods:
            chain = ["w4", f"{method}_0.5", f"{method}_1"]
            ax.plot([v(c, "p_target") for c in chain], [v(c, key) for c in chain], color=color, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.2)
        ax.plot([v("scalar_1", "p_target"), v("w1", "p_target")], [v("scalar_1", key), v("w1", key)], color="#2a78d6", lw=1, ls=":")
        for c, mark in (("w4", "s"), ("w1", "D"), ("wrongclass_1", "x")):
            ax.scatter([v(c, "p_target")], [v(c, key)], color=INK if c != "wrongclass_1" else MUTED, marker=mark, s=36, zorder=4)
            if c != "wrongclass_1":
                ax.annotate({"w4": "w = 4", "w1": "w = 1"}[c], (v(c, "p_target"), v(c, key)), xytext=(-30 if c == "w4" else 6, 6),
                            textcoords="offset points", color=MUTED, fontsize=7.5)
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=0, labelsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("fidelity: classifier p(target class)", color=MUTED, fontsize=9.5)
        ax.set_ylabel(label, color=MUTED, fontsize=9.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=5, label=n) for _, c, n in methods]
    handles.append(Line2D([], [], color=MUTED, marker="x", ls="", label="score with wrong class's subspace (λ = 1)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8)
    header(fig, "At equal fidelity, score-space damping recovers layout and condition-space recovers semantics",
           "Held-out seeds 16–31, 8 classes. Points along each line: λ = 0 (w = 4), 0.5, 1. Up and to the right is better. All conditions: 98 model passes.", 0.915)
    save(fig, out, "13_intervention_tradeoff.png", (0, 0.06, 1, 0.87))

    conditions = ["w1", "w4", "scalar_1", "score_1", "freq_1", "cond_1", "feat_1", "wrongclass_1", "cond_0.5"]
    tile, pad = 110, 92
    sheet = Image.new("RGB", (pad + 8 * tile, len(conditions) * tile), "white")
    draw = ImageDraw.Draw(sheet)
    for r, c in enumerate(conditions):
        image = Image.open(run / "samples" / f"{c}_seed_{sheet_seed}.png").convert("RGB")
        size = (image.size[0] - 2) // 4
        draw.text((4, r * tile + tile // 2 - 6), c, fill="black")
        for i in range(8):
            a, b = divmod(i, 4)
            sheet.paste(image.crop((2 + b * size, 2 + a * size, 2 + b * size + 256, 2 + a * size + 256)).resize((tile - 2, tile - 2)), (pad + i * tile, r * tile + 1))
    sheet.save(out / f"14_intervention_samples_seed{sheet_seed}.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--embed", action="store_true")
    p.add_argument("--stats", action="store_true")
    p.add_argument("--figures", action="store_true")
    p.add_argument("--sheet-seed", type=int, default=23)
    a = p.parse_args()
    run = Path(a.run)
    if a.embed:
        embed(run)
    if a.stats:
        for row in stats(run):
            print(row["condition"].ljust(13), {k: round(row[k], 3) for k in ("V_low", "dino", "clip", "p_target", "top1", "retention")}, flush=True)
    if a.figures:
        figures(run, a.sheet_seed)


if __name__ == "__main__":
    main()
