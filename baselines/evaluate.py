"""Decode, embed and score a sample.py run: final diversity and quality per method, and diversity over sampling time.

usage: python evaluate.py --run runs/main [--grid-class 0] [--time-dino]
Writes into the run folder:
  metrics.csv        per method: within-class diversity (Vendi, MSS on DINOv2; MSS on CLIP; layout spread V_low)
                     and quality (classifier confidence in the target class, top-1), with 90% subsampling intervals
  over_time.csv      per method and recorded step: scale-free spread V of x_hat (all frequencies and low frequencies),
                     plus DINOv2 MSS of decoded x_hat with --time-dino
  tradeoff.png       diversity (Vendi) against quality (target-class confidence)
  over_time.png      spread of x_hat over t
  samples.png        with --grid-class c: one row of images of class c per method
Embeddings are cached next to each batch (embed_seed_<n>.npz), so rerunning is cheap.
Every diversity number is computed within a class (across seeds) and then averaged over the classes.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import LABEL_NAMES, LABELS, default_device

VAE = "stabilityai/sd-vae-ft-mse"  # SiT's sample.py default
LATENT_SCALE = 0.18215
DINO = ("vit_base_patch14_dinov2.lvd142m", dict(num_classes=0, img_size=224))
CLIP = ("vit_base_patch16_clip_224.openai", dict(num_classes=0))
CLASSIFIER = ("vit_base_patch16_224.augreg2_in21k_ft_in1k", {})
LOW_CUTOFF = 4  # radial frequency (cycles per 32-pixel latent) below which a pattern counts as layout
N_SUB, SUB_FRAC = 500, 0.8


# ---------------------------------------------------------------- statistics

def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def vendi(feats):
    """exp(entropy of the eigenvalues of K/n), K = cosine similarity: the effective number of distinct samples, 1..n."""
    u = unit(feats.astype(np.float64))
    lam = np.linalg.eigvalsh(u @ u.T / len(u))
    lam = lam[lam > 1e-12]
    return float(np.exp(-(lam * np.log(lam)).sum()))


def mss(feats):
    """Mean cosine similarity over distinct pairs (lower = more diverse)."""
    u = unit(feats.astype(np.float64))
    n = len(u)
    return float(((u @ u.T).sum() - n) / (n * (n - 1)))


def spread_v(x):
    """Scale-free spread: variance / mean squared norm = 1 - ||mean||^2 / mean ||x_i||^2. x: (samples, dim)."""
    x = x.astype(np.float64)
    return float(1 - (x.mean(0) ** 2).sum() / (x ** 2).sum(1).mean())


def low_band(latents):
    """Low-pass (radius <= LOW_CUTOFF) of latents (..., 4, 32, 32), flattened."""
    f = np.fft.fftfreq(32, 1 / 32)
    mask = np.hypot(*np.meshgrid(f, f, indexing="ij")) <= LOW_CUTOFF
    low = np.fft.ifft2(np.fft.fft2(latents.astype(np.float64), axes=(-2, -1)) * mask, axes=(-2, -1)).real
    return low.reshape(*latents.shape[:-3], -1)


# ---------------------------------------------------------------- models

def load_encoders(device):
    import timm
    from diffusers.models import AutoencoderKL

    def encoder(name, kwargs):
        model = timm.create_model(name, pretrained=True, **kwargs).to(device).eval()
        cfg = timm.data.resolve_data_config({}, model=model)
        mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
        # the model's own input size (DINOv2's pretrained config says 518 even when built with img_size=224)
        size = model.patch_embed.img_size[0] if hasattr(model, "patch_embed") else cfg["input_size"][-1]

        def run(images):  # images in [0, 1], (n, 3, 256, 256)
            # antialiased resize runs on CPU: MPS does not implement it
            small = F.interpolate(images.cpu(), size=size, mode="bicubic", antialias=True).clamp(0, 1).to(device)
            return model((small - mean) / std).float()
        return run

    vae = AutoencoderKL.from_pretrained(VAE).to(device).eval()
    decode = lambda z: ((vae.decode(z / LATENT_SCALE).sample + 1) / 2).clamp(0, 1)
    return decode, encoder(*DINO), encoder(*CLIP), encoder(*CLASSIFIER)


@torch.no_grad()
def embed_run(run, device, time_dino):
    models = None
    labels = torch.tensor(LABELS, device=device)
    for path in sorted(run.glob("*/seed_*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        target = path.with_name(path.name.replace("seed_", "embed_seed_"))
        if target.exists() and (not time_dino or "dino_t" in np.load(target).files):
            continue
        models = models or load_encoders(device)
        decode, dino, clip, classifier = models
        batch = np.load(path)
        images = decode(torch.from_numpy(batch["final"]).to(device))
        prob = classifier(images).softmax(-1)
        out = dict(dino=dino(images).cpu().numpy(), clip=clip(images).cpu().numpy(),
                   p_target=prob[torch.arange(len(labels)), labels].cpu().numpy(),
                   top1=(prob.argmax(-1) == labels).cpu().numpy())
        if time_dino:
            out["dino_t"] = np.stack([dino(decode(torch.from_numpy(x.astype(np.float32)).to(device))).cpu().numpy()
                                      for x in batch["xhat"]])
        tmp = target.with_name(target.stem + ".tmp.npz")
        np.savez(tmp, **out)
        tmp.rename(target)
        print("embedded", path.parent.name, path.stem, flush=True)


def load_run(run):
    """{method: dict of arrays with seeds on axis 0}, restricted to seeds present for every method."""
    found = {d.name: {int(p.stem.split("_")[1]) for p in d.glob("seed_*.npz") if not p.name.endswith(".tmp.npz")}
             for d in sorted(run.iterdir()) if d.is_dir()}
    found = {k: v for k, v in found.items() if v}
    seeds = sorted(set.intersection(*found.values()))
    data = {}
    for method in found:
        batches = [np.load(run / method / f"seed_{s}.npz") for s in seeds]
        embeds = [np.load(run / method / f"embed_seed_{s}.npz") for s in seeds]
        data[method] = dict(final=np.stack([b["final"] for b in batches]), xhat=np.stack([b["xhat"] for b in batches]),
                            t=batches[0]["t"], **{k: np.stack([e[k] for e in embeds]) for k in embeds[0].files})
    return data, seeds


# ---------------------------------------------------------------- metrics

def final_metrics(d, idx):
    classes = range(len(LABELS))
    return dict(
        vendi_dino=np.mean([vendi(d["dino"][idx, c]) for c in classes]),
        mss_dino=np.mean([mss(d["dino"][idx, c]) for c in classes]),
        mss_clip=np.mean([mss(d["clip"][idx, c]) for c in classes]),
        layout_V_low=np.mean([spread_v(low_band(d["final"][idx, c])) for c in classes]),
        p_target=float(d["p_target"][idx].mean()), top1=float(d["top1"][idx].mean()))


def over_time(d):
    rows = []
    for k, t in enumerate(d["t"]):
        x = d["xhat"][:, k].astype(np.float64)  # (seeds, classes, 4, 32, 32)
        row = dict(step=k, t=float(t),
                   V_all=np.mean([spread_v(x[:, c].reshape(len(x), -1)) for c in range(x.shape[1])]),
                   V_low=np.mean([spread_v(low_band(x[:, c])) for c in range(x.shape[1])]))
        if "dino_t" in d:
            row["mss_dino"] = np.mean([mss(d["dino_t"][:, k, c]) for c in range(x.shape[1])])
        rows.append(row)
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- figures

def cfg_order(names):
    sweep = sorted((n for n in names if n.startswith("cfg_w")), key=lambda n: float(n[5:].split(":")[0]))
    return sweep, [n for n in names if n not in sweep]


def plot_tradeoff(rows, path):
    import matplotlib.pyplot as plt
    by = {r["method"]: r for r in rows}
    sweep, others = cfg_order(list(by))
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    if sweep:
        ax.plot([by[m]["p_target"] for m in sweep], [by[m]["vendi_dino"] for m in sweep], "-o", color="0.5",
                label="CFG w-sweep", zorder=1)
        for m in sweep:
            ax.annotate(m[4:], (by[m]["p_target"], by[m]["vendi_dino"]), fontsize=7, color="0.4",
                        xytext=(4, -10), textcoords="offset points")
    for m in others:
        r = by[m]
        ax.errorbar(r["p_target"], r["vendi_dino"], xerr=[[r["p_target"] - r["p_target_lo"]], [r["p_target_hi"] - r["p_target"]]],
                    yerr=[[r["vendi_dino"] - r["vendi_dino_lo"]], [r["vendi_dino_hi"] - r["vendi_dino"]]], fmt="o", capsize=2)
        ax.annotate(m, (r["p_target"], r["vendi_dino"]), fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("quality: classifier confidence in target class")
    ax.set_ylabel("diversity: within-class Vendi (DINOv2)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


def plot_over_time(rows, path):
    import matplotlib.pyplot as plt
    keys = ["V_all", "V_low"] + (["mss_dino"] if "mss_dino" in rows[0] else [])
    titles = dict(V_all="spread of x̂ (all frequencies)", V_low="spread of x̂ (layout, low frequencies)",
                  mss_dino="DINOv2 similarity of decoded x̂ (lower = more diverse)")
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.8), dpi=150)
    names = list(dict.fromkeys(r["method"] for r in rows))
    sweep = sorted((m for m in names if m.startswith("cfg_w")), key=lambda m: float(m[5:]))
    others = [m for m in names if m not in sweep]
    # CFG sweep in greys (light = weak w), other methods in distinct colours
    style = {m: dict(color=str(0.75 - 0.6 * i / max(1, len(sweep) - 1)), ls="--") for i, m in enumerate(sweep)}
    colours = [plt.cm.tab10(i) for i in (0, 1, 2, 3, 4, 5, 6, 8, 9)]  # skip tab10 grey
    style.update({m: dict(color=colours[i % len(colours)], ls="-") for i, m in enumerate(others)})
    for ax, key in zip(np.atleast_1d(axes), keys):
        for m in sweep + others:
            sel = [r for r in rows if r["method"] == m]
            ax.plot([r["t"] for r in sel], [r[key] for r in sel], marker="o", ms=2.5, lw=1.2, label=m, **style[m])
        ax.set_title(titles[key], fontsize=9)
        ax.set_xlabel("t (0 = noise, 1 = data)")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)


@torch.no_grad()
def plot_samples(run, data, seeds, c, device, path, per_row=8):
    import matplotlib.pyplot as plt
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE).to(device).eval()
    rows = []
    for m, d in data.items():
        z = torch.from_numpy(d["final"][:per_row, c]).to(device)
        img = ((vae.decode(z / LATENT_SCALE).sample + 1) / 2).clamp(0, 1)
        img = F.interpolate(img.cpu(), size=128, mode="bilinear", antialias=True)
        rows.append(np.concatenate(list(img.permute(0, 2, 3, 1).cpu().numpy()), axis=1))
    fig, ax = plt.subplots(figsize=(per_row * 1.2, len(rows) * 1.25), dpi=150)
    ax.imshow(np.concatenate(rows, axis=0))
    ax.set_yticks([64 + 128 * i for i in range(len(rows))], list(data))
    ax.set_xticks([])
    ax.set_title(f"{LABEL_NAMES[c]}, seeds {seeds[:per_row]}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--time-dino", action="store_true", help="also decode and embed x_hat at the recorded steps (10x cost)")
    p.add_argument("--grid-class", type=int, default=None, help="class index (0-7) for samples.png")
    a = p.parse_args()

    torch.set_grad_enabled(False)
    run, device = Path(a.run), a.device or default_device()
    embed_run(run, device, a.time_dino)
    data, seeds = load_run(run)
    print(f"{len(data)} methods, {len(seeds)} shared seeds")

    rng = np.random.default_rng(0)
    n_sub = max(2, int(round(SUB_FRAC * len(seeds))))
    subsets = [np.sort(rng.choice(len(seeds), n_sub, replace=False)) for _ in range(N_SUB)]
    rows = []
    for m, d in data.items():
        point = final_metrics(d, np.arange(len(seeds)))
        subs = [final_metrics(d, idx) for idx in subsets]
        row = dict(method=m, n_seeds=len(seeds), **point)
        for key in point:
            # Vendi and spread depend on sample size, so subsample values are biased; keep the spread of the
            # subsample distribution but centre it on the full-sample estimate.
            lo, med, hi = np.percentile([s[key] for s in subs], [5, 50, 95])
            row[f"{key}_lo"], row[f"{key}_hi"] = point[key] - (med - lo), point[key] + (hi - med)
        rows.append(row)
    write_csv(run / "metrics.csv", rows)
    time_rows = [dict(method=m, **r) for m, d in data.items() for r in over_time(d)]
    write_csv(run / "over_time.csv", time_rows)
    plot_tradeoff(rows, run / "tradeoff.png")
    plot_over_time(time_rows, run / "over_time.png")
    if a.grid_class is not None:
        plot_samples(run, data, seeds, a.grid_class, device, run / "samples.png")

    width = max(len(r["method"]) for r in rows)
    print(f"{'method':{width}s}  vendi  mss_dino  mss_clip  V_low  p_target  top1")
    for r in rows:
        print(f"{r['method']:{width}s}  {r['vendi_dino']:5.2f}  {r['mss_dino']:8.3f}  {r['mss_clip']:8.3f}  "
              f"{r['layout_V_low']:5.3f}  {r['p_target']:8.3f}  {r['top1']:4.2f}")


if __name__ == "__main__":
    main()
