"""Decode x_hat at every step, plus the final latents, and store DINOv2 and CLIP global embeddings for each image.

Writes <run>/analysis/embeddings/w<w>_seed_<n>.npz with dino (50, 8, 768) and clip (50, 8, 512): rows 0-48 are x_hat at each
model call and row 49 is the final sample. Decoded finals go to <run>/analysis/samples/ as grids. Finished files are skipped.
usage: guidance_embed.py --run <dir> [--seeds ...] [--ws 4 1] [--device mps]
"""
import argparse, json
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

DINO = "vit_base_patch14_dinov2.lvd142m"
CLIP = "vit_base_patch16_clip_224.openai"
VAE = "stabilityai/sd-vae-ft-mse"
LATENT_SCALE = 0.18215  # same constant as sample.py
SIZE = 224
BATCH = 8


def encoder(name, device, **kwargs):
    model = timm.create_model(name, pretrained=True, **kwargs).to(device).eval()
    cfg = timm.data.resolve_model_data_config(model)
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    return model, mean, std


def embed(enc, images):
    """images: (B, 3, 224, 224) in [0, 1]. Returns the model's global embedding."""
    model, mean, std = enc
    return model((images - mean) / std).float().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(16)))
    p.add_argument("--ws", type=float, nargs="+", default=[4.0, 1.0])
    p.add_argument("--device", default="mps")
    p.add_argument("--final-only", action="store_true", help="embed only the final samples; arrays then have one row")
    a = p.parse_args()
    torch.set_grad_enabled(False)

    run = Path(a.run)
    out, samples = run / "analysis" / "embeddings", run / "analysis" / "samples"
    out.mkdir(parents=True, exist_ok=True)
    samples.mkdir(exist_ok=True)
    vae = AutoencoderKL.from_pretrained(VAE).to(a.device).eval()
    # img_size=224 resamples DINOv2's position embeddings from its native 518, giving 16x16 patches of size 14.
    dino = encoder(DINO, a.device, num_classes=0, img_size=SIZE)
    clip = encoder(CLIP, a.device)  # the default head is CLIP's learned projection, so this is the 512-d image embedding
    info = dict(vae=VAE, dino=DINO, clip=CLIP, image_size=SIZE, timm=timm.__version__, rows="0-48 x_hat per model call, 49 final",
                dino_hub=dino[0].pretrained_cfg.get("hf_hub_id"), clip_hub=clip[0].pretrained_cfg.get("hf_hub_id"))
    (out / "embeddings.json").write_text(json.dumps(info, indent=2))

    for seed in a.seeds:
        for w in a.ws:
            path = out / f"w{w:g}_seed_{seed}.npz"
            if path.exists():
                continue
            d = np.load(run / f"w{w:g}" / f"seed_{seed}" / "trajectory.npz")
            latents = d["final"][None] if a.final_only else np.concatenate([d["x_hat"], d["final"][None]], 0)  # (rows, 8, 4, 32, 32)
            rows = dict(dino=[], clip=[])
            for k in range(len(latents)):
                z = torch.from_numpy(latents[k]).to(a.device)
                image = ((vae.decode(z / LATENT_SCALE).sample + 1) / 2).clamp(0, 1)
                if k == len(latents) - 1:
                    save_image(image.cpu(), samples / f"w{w:g}_seed_{seed}.png", nrow=4)
                small = F.interpolate(image, size=SIZE, mode="bicubic", antialias=True).clamp(0, 1)
                rows["dino"].append(embed(dino, small))
                rows["clip"].append(embed(clip, small))
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, **{k: np.stack(v) for k, v in rows.items()})
            tmp.rename(path)  # a file at the final path is always complete
            print(f"w={w:g} seed={seed} done", flush=True)


if __name__ == "__main__":
    main()
