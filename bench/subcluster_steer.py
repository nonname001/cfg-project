"""Steer each sample to a real-image subcluster of its class instead of letting guidance pull it to the class mode.

Subclusters: cosine k-means centroids of the class's real training images in DINOv2 space (subclusters.py, K = 8). Each sample gets
a target subcluster k, drawn from the real subcluster shares (seeded per seed and class, so every strength uses the same targets).
While s_t is in [STEER_LO, STEER_HI), every step adds a push to the guided velocity:
  x_hat = x_t + (1 - t) v                                   the predicted clean latent
  g     = d cos(DINOv2(decode(x_hat)), centroid_k) / d x_hat  through the VAE decoder and DINOv2 (not SiT)
  g_low = low band (<= 4 cycles) of g, normalized            the band that carries the semantic decision at this time (patching)
  v    += s * ||Delta|| * g_low                              Delta = cond - uncond at this step, per sample
Outside the window: plain w = 4. Condition name steer_<s>; steer_0 reproduces w = 4 and records the targets.
Options: --lo/--hi move the steering window (name gains _<lo>-<hi>); --guide-from t uses w = 1 before s_t = t (the guidance
interval; name gains _g<t>). With the defaults the names and samples are unchanged.
Output as guidance_intervene.py (<out>/<condition>/seed_<n>.npz with "final"), plus "target" (8,).
usage: subcluster_steer.py --weights <file> --out <dir> --seeds 16 ... --scales 0 1 2 [--k 8]
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "bench")
from common import GUIDED_CHANNELS, LABELS, NULL_CLASS, array_sha256, initial_noise, load_model
from transport import Sampler, create_transport
from guidance_embed import decode, encode, load_models
from patch_sweep import band_mask, band_part

W, STEER_LO, STEER_HI = 4.0, 0.06, 0.3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True); p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True); p.add_argument("--scales", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    p.add_argument("--k", type=int, default=8); p.add_argument("--device", default="mps")
    p.add_argument("--lo", type=float, default=STEER_LO); p.add_argument("--hi", type=float, default=STEER_HI)
    p.add_argument("--guide-from", type=float, default=0.0)
    a = p.parse_args()
    torch.set_grad_enabled(False)
    dev, n = a.device, len(LABELS)
    tab = np.load(f"outputs/guidance_delta/analysis/subclusters_k{a.k}.npz")
    cent = torch.from_numpy(np.stack([tab[f"centroids_{l}"] for l in LABELS])).float().to(dev)  # (8 classes, K, 768)
    share = np.stack([tab[f"share_{l}"] for l in LABELS])
    model = load_model("SiT-L/2", a.weights, dev)
    vae, dino, _ = load_models(dev)
    for m in (vae, dino[0]):
        for prm in m.parameters():
            prm.requires_grad_(False)
    low = band_mask("low", dev)
    sample_fn = Sampler(create_transport("Linear", "velocity", None, None, None)).sample_ode(
        sampling_method="euler", num_steps=50, atol=1e-6, rtol=1e-3, reverse=False)
    out = Path(a.out)
    for seed in a.seeds:
        rng = np.random.default_rng(seed + 31337)
        target = np.array([rng.choice(a.k, p=share[i] / share[i].sum()) for i in range(n)])
        tc = cent[torch.arange(n), torch.from_numpy(target).to(dev)]  # (8, 768) target centroid per sample
        for s in a.scales:
            name = f"steer_{s:g}" + (f"_{a.lo:g}-{a.hi:g}" if (a.lo, a.hi) != (STEER_LO, STEER_HI) else "") + (f"_g{a.guide_from:g}" if a.guide_from else "")
            path = out / name / f"seed_{seed}.npz"
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            last = {}
            fwd = model.forward

            def recording(xb, tb, yb):
                last["out"] = fwd(xb, tb, yb)
                return last["out"]

            def guided(xb, tb, y, cfg_scale):
                t = float(tb[0])
                v = model.forward_with_cfg(xb, tb, y, 1.0 if t < a.guide_from else cfg_scale)
                if s == 0 or not a.lo <= t < a.hi:
                    return v
                raw = last["out"]
                delta = (raw[:n, :GUIDED_CHANNELS] - raw[n:, :GUIDED_CHANNELS]).float()
                xhat = (xb[:n] + (1 - t) * v[:n]).float()
                grads = []
                with torch.enable_grad():
                    for i in range(n):  # one image at a time: the decoder's backward needs ~2.3 GB per image (9 GB for 4)
                        lat = xhat[i:i + 1].detach().clone().requires_grad_(True)
                        f = encode(dino, decode(vae, lat)[1]).float()
                        cos = (f / f.norm(dim=1, keepdim=True) * tc[i:i + 1]).sum()
                        cos.backward()
                        grads.append(lat.grad.detach())
                        del lat, f, cos
                torch.mps.empty_cache()  # hand the backward pass's buffers back before the next SiT step
                g = band_part(torch.cat(grads), low)
                g = g / g.reshape(n, -1).norm(dim=1).clamp_min(1e-12).reshape(n, 1, 1, 1)
                push = s * delta.reshape(n, -1).norm(dim=1).reshape(n, 1, 1, 1) * g
                v = v.clone()
                v[:n] += push.to(v.dtype); v[n:] += push.to(v.dtype)  # both halves stay identical, as in forward_with_cfg
                return v

            z = initial_noise(seed)
            x = torch.cat([z, z], 0).to(dev)
            y = torch.tensor(LABELS + [NULL_CLASS] * n, device=dev)
            model.forward = recording
            start = time.perf_counter()
            try:
                final = sample_fn(x, guided, y=y, cfg_scale=W)[-1][:n].float().cpu().numpy()
            finally:
                del model.__dict__["forward"]
            assert np.isfinite(final).all()
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, seed=seed, final=final, final_sha256=array_sha256(final), target=target, scale=s)
            tmp.rename(path)
            print(f"{name} seed={seed}: {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
