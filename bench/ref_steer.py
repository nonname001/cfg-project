"""Steer each sample toward a real reference image's coarse layout (ILVR-style), as a way to reach real subclusters.

Patching showed the guided semantic outcome at s_t ~0.24 sits mostly in the low band (<= 4 cycles) of the state x_t. So instead of a
feature gradient, move that band toward a real image's: each sample gets a random validation image of its class as reference
(its subcluster, subclusters.py, is the target, so targets follow the validation images' subcluster frequencies). At one step k:
  ref_t = t z_ref + (1 - t) eps       the reference's state on SiT's straight path, with the sample's own initial noise eps
  x_t  += alpha * (low(ref_t) - low(x_t))
then sampling continues at w = 4 (or w = 1 before --guide-from). Condition ref_<alpha>_k<k>[_g<t>]; ref_0 is w = 4 with the
same references (the paired baseline). Writes "final", "ref" (index of the reference among the class's validation images) and
"target" (its subcluster).
--encode: VAE-encode the 400 validation images (posterior mean, as SiT's latents) -> outputs/guidance_delta/analysis/val_latents.npz
usage: ref_steer.py --encode
       ref_steer.py --weights <file> --out <dir> --seeds 16 ... --alphas 0 0.5 1 --ks 6 12 [--guide-from 0.12]
"""
import argparse, io, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "bench")
from common import LABELS, NULL_CLASS, array_sha256, initial_noise, load_model
from transport import Sampler, create_transport
from patch_sweep import band_mask, band_part

LAT = Path("outputs/guidance_delta/analysis/val_latents.npz")


def encode_val(device):
    from PIL import Image
    from guidance_embed import LATENT_SCALE, load_models
    from real_fidelity import center_crop_arr
    vae, _, _ = load_models(device)
    real = Path("outputs/real_imagenet_val")
    out = {}
    for l in LABELS:
        files = sorted((real / str(l)).glob("*"))  # same order as real_fidelity.embed, so indices match dino.npz
        arr = np.stack([center_crop_arr(Image.open(io.BytesIO(f.read_bytes())).convert("RGB")) for f in files])
        z = [(vae.encode(torch.from_numpy(arr[i:i + 10]).permute(0, 3, 1, 2).float().div(255).to(device) * 2 - 1).latent_dist.mean
              * LATENT_SCALE).cpu().numpy() for i in range(0, len(arr), 10)]
        out[f"z_{l}"] = np.concatenate(z).astype(np.float32)
    np.savez(LAT, **out)
    print("encoded", {l: out[f"z_{l}"].shape[0] for l in LABELS})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encode", action="store_true"); p.add_argument("--weights"); p.add_argument("--out")
    p.add_argument("--seeds", type=int, nargs="+"); p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    p.add_argument("--ks", type=int, nargs="+", default=[6, 12]); p.add_argument("--guide-from", type=float, default=0.0)
    p.add_argument("--k-sub", type=int, default=8); p.add_argument("--device", default="mps")
    a = p.parse_args()
    torch.set_grad_enabled(False)
    if a.encode:
        encode_val(a.device); return
    dev, n = a.device, len(LABELS)
    lat = np.load(LAT); sub = np.load(f"outputs/guidance_delta/analysis/subclusters_k{a.k_sub}.npz")
    model = load_model("SiT-L/2", a.weights, dev)
    sample_fn = Sampler(create_transport("Linear", "velocity", None, None, None)).sample_ode(
        sampling_method="euler", num_steps=50, atol=1e-6, rtol=1e-3, reverse=False)
    low = band_mask("low", dev)
    out = Path(a.out)
    plan = [(0.0, None)] + [(al, k) for al in a.alphas if al > 0 for k in a.ks]
    for seed in a.seeds:
        rng = np.random.default_rng(seed + 4242)
        ref = np.array([rng.integers(len(lat[f"z_{l}"])) for l in LABELS])
        target = np.array([sub[f"val_sub_{l}"][ref[i]] for i, l in enumerate(LABELS)])
        z_ref = torch.from_numpy(np.stack([lat[f"z_{l}"][ref[i]] for i, l in enumerate(LABELS)])).to(dev)
        z = initial_noise(seed)
        for alpha, k in plan:
            name = "ref_0" if k is None else f"ref_{alpha:g}_k{k}"
            name += f"_g{a.guide_from:g}" if a.guide_from and k is not None else ""
            path = out / name / f"seed_{seed}.npz"
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            eps = z.to(dev)

            def guided(xb, tb, y, cfg_scale):
                t = float(tb[0])
                return model.forward_with_cfg(xb, tb, y, 1.0 if (k is not None and t < a.guide_from) else cfg_scale)

            start = time.perf_counter()
            y = torch.tensor(LABELS + [NULL_CLASS] * n, device=dev)
            solver = sample_fn.__self__; full = solver.t
            if k is None:
                final = sample_fn(torch.cat([eps, eps]), guided, y=y, cfg_scale=4.0)[-1][:n]
            else:
                # Sample to step k, patch the low band, then continue from step k (the grid is cut as in switch_sweep).
                solver.t = full[:k + 1]
                try:
                    x_k = sample_fn(torch.cat([eps, eps]), guided, y=y, cfg_scale=4.0)[-1][:n]
                finally:
                    solver.t = full
                t = float(full[k])
                ref_t = t * z_ref + (1 - t) * eps
                x_k = x_k + alpha * (band_part(ref_t, low) - band_part(x_k, low))
                solver.t = full[k:]
                try:
                    final = sample_fn(torch.cat([x_k, x_k]), guided, y=y, cfg_scale=4.0)[-1][:n]
                finally:
                    solver.t = full
            final = final.float().cpu().numpy()
            assert np.isfinite(final).all()
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, seed=seed, final=final, final_sha256=array_sha256(final), ref=ref, target=target, alpha=alpha, k=-1 if k is None else k)
            tmp.rename(path)
            print(f"{name} seed={seed}: {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
