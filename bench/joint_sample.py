"""Joint sampling with repulsion between samples of a class (particle-guidance style), saving final latents per seed.

Every other method in this project changes each sample on its own. Here all seeds of a group are sampled together (one
batch of seeds x 8 classes), and while s_t < REP_END each sample is pushed away from the other samples of its class:
  x_hat_i  = x_t + (1 - t) v_i, the predicted clean latent of sample i (straight path)
  k_ij     = exp(-||x_hat_i - x_hat_j||^2 / h), h = median pairwise squared distance / log(S) (median heuristic, per class)
  r_i      = sum_j k_ij (x_hat_i - x_hat_j), the negative gradient direction of sum_j k_ij (Corso et al. 2023)
  v_i     += s * ||Delta_i|| * r_i / ||r_i||, so s = 1 pushes as hard as the sample's own guidance vector Delta
on all 4 latent channels. Diversity is then measured over exactly the seeds that repelled each other.

Condition names join parts with '+': rep_<s> (repulsion strength), w_<w> (guidance scale at every step, default 4),
score_<L> (score damping during s_t < 0.3, as guidance_intervene.py), repend_<t> (repulsion window end, default 0.3),
frep_<s> with frepblock_<b> (default 12): the same repulsion, but between the class's samples' conditional-branch token
activations at block b (a semantic feature space; see cond_tokens.py), added to those activations with size s * ||d_i||,
d_i = cond - uncond output of block b for sample i. frep needs every seed of a class in one forward pass, so with frep the
batch is chunked by class instead of by rows.
drep_<s>: the same repulsion in DINOv2 space: each window step decodes x_hat with the VAE, embeds it with DINOv2 (exactly as
guidance_embed.py), and pushes x_hat_i along -dU/dx_hat_i, U = sum_ij k_ij on the DINOv2 features (gradient through the
decoder and DINOv2, not SiT), normalized and scaled by s * ||Delta_i||. Caution: DINO diversity is measured in the same
feature space, so CLIP diversity is the independent check; precision also uses DINOv2 features.
w_4 alone is plain CFG, the joint-sampling control (compare with the per-seed w4 samples to check batching effects).
Output: <out>/<condition>/seed_<n>.npz with the same keys as guidance_intervene.py, so guidance_intervene_eval.py applies.
usage: joint_sample.py --tables <file> --weights <file> --out <dir> --seeds 16 ... 31 --conditions rep_1 ...
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "bench")
from common import GUIDED_CHANNELS, LABELS, NULL_CLASS, array_sha256, initial_noise, load_model, sha256
from transport import Sampler, create_transport

W = 4.0
EARLY_END = 0.3
CHUNK = 32  # samples per forward pass (64 rows with the null branch)


def parse(condition):
    return {k: float(v) for k, v in (part.split("_") for part in condition.split("+"))}


def dino_repulsion(xhat, embedders, chunk=4):
    """xhat: (S, 4, 32, 32) predicted clean latents of one class. Returns -dU/dxhat, U = sum_ij exp(-||f_i - f_j||^2 / h) on the
    DINOv2 features f of the decoded images (h by the median heuristic). The feature gradient is computed for all S at once,
    then pulled back through the VAE decoder and DINOv2 in chunks to bound memory."""
    from guidance_embed import decode, encode
    vae, dino = embedders
    S = len(xhat)
    with torch.no_grad():
        feats = torch.cat([encode(dino, decode(vae, xhat[i:i + chunk])[1]) for i in range(0, S, chunk)]).float()
    with torch.enable_grad():
        f = feats.clone().requires_grad_(True)
        d2 = torch.cdist(f, f) ** 2
        h = d2.detach()[~torch.eye(S, dtype=bool, device=f.device)].median() / np.log(S)
        torch.exp(-d2 / h).sum().backward()
        g_feat = f.grad
        grads = []
        for i in range(0, S, chunk):
            lat = xhat[i:i + chunk].detach().clone().requires_grad_(True)
            encode(dino, decode(vae, lat)[1]).float().backward(g_feat[i:i + chunk])
            grads.append(lat.grad)
    return -torch.cat(grads)


def run_group(model, sample_fn, condition, seeds, device, latent_basis, embedders=None):
    parts = parse(condition)
    assert set(parts) <= {"rep", "w", "score", "repend", "frep", "frepblock", "drep"}, f"unknown part in {condition}"
    w, s_rep, rep_end = parts.get("w", W), parts.get("rep", 0.0), parts.get("repend", EARLY_END)
    n_cls, S = len(LABELS), len(seeds)
    z = torch.cat([initial_noise(s) for s in seeds], 0)  # row s * 8 + c: seed seeds[s], class c
    N = len(z)
    y = torch.tensor(LABELS * S, device=device)
    null = torch.full((CHUNK,), NULL_CLASS, device=device)
    basis = latent_basis.repeat(S, 1, 1) if "score" in parts else None  # (N, 3, 3072), row's class basis

    state = dict(t=1.0)

    def feature_repulsion(_module, _inputs, out):
        # frep: out is (2k, tokens, width) for one class's k = S seeds (conditional rows first). Push the conditional rows
        # apart in this block's token-activation space, scaled to each sample's class push at this block.
        if state["t"] >= rep_end:
            return None
        k = out.shape[0] // 2
        h = out[:k].float().reshape(k, -1)
        d2 = torch.cdist(h, h) ** 2
        bw = d2[~torch.eye(k, dtype=bool, device=out.device)].median() / np.log(k)
        r = (torch.exp(-d2 / bw)[:, :, None] * (h[:, None] - h[None])).sum(1)
        r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-12)
        scale = (out[:k] - out[k:]).float().reshape(k, -1).norm(dim=1, keepdim=True)
        return torch.cat([out[:k] + (parts["frep"] * scale * r).reshape(out[:k].shape).to(out.dtype), out[k:]], 0)

    def guided(xb, tb, **_):
        x, t = xb[:N], float(tb[0])
        state["t"] = t
        cond, unc = torch.empty_like(x), torch.empty_like(x)
        if "frep" in parts:
            groups = [torch.arange(c, N, n_cls, device=device) for c in range(n_cls)]  # all seeds of one class per pass
        else:
            groups = [torch.arange(i, min(i + CHUNK, N), device=device) for i in range(0, N, CHUNK)]
        for idx in groups:
            xc, k = x[idx], len(idx)
            out = model.forward(torch.cat([xc, xc]), torch.full((2 * k,), t, device=device), torch.cat([y[idx], null[:k] if k <= CHUNK else torch.full((k,), NULL_CLASS, device=device)]))
            cond[idx], unc[idx] = out[:k], out[k:]
        delta = (cond[:, :GUIDED_CHANNELS] - unc[:, :GUIDED_CHANNELS]).float()
        eps = unc[:, :GUIDED_CHANNELS] + w * (cond[:, :GUIDED_CHANNELS] - unc[:, :GUIDED_CHANNELS])
        v = torch.cat([eps, cond[:, GUIDED_CHANNELS:]], 1)
        if "score" in parts and t < EARLY_END:
            flat = delta.reshape(N, -1)
            part = torch.einsum("nk,nkd->nd", torch.einsum("nkd,nd->nk", basis, flat), basis).reshape(delta.shape)
            v = v.clone()
            v[:, :GUIDED_CHANNELS] -= (w - 1) * parts["score"] * part
        if s_rep and t < rep_end:
            xhat = (x + (1 - t) * v).float().reshape(S, n_cls, -1)  # (seeds, classes, dim)
            push = torch.zeros_like(xhat)
            for c in range(n_cls):
                p = xhat[:, c]  # (S, dim)
                d2 = torch.cdist(p, p) ** 2
                h = d2[~torch.eye(S, dtype=bool, device=device)].median() / np.log(S)
                k = torch.exp(-d2 / h)
                r = (k[:, :, None] * (p[:, None] - p[None])).sum(1)  # (S, dim); diagonal terms are zero
                push[:, c] = r / r.norm(dim=1, keepdim=True).clamp_min(1e-12)
            scale = delta.reshape(N, -1).norm(dim=1).reshape(S, n_cls, 1)  # ||Delta_i||
            v = v + (s_rep * scale * push).reshape(v.shape).to(v.dtype)
        if parts.get("drep") and t < rep_end:
            xhat = (x + (1 - t) * v).float()
            push = torch.zeros_like(xhat)
            for c in range(n_cls):
                idx = torch.arange(c, N, n_cls, device=device)
                r = dino_repulsion(xhat[idx], embedders).reshape(len(idx), -1)
                push[idx] = (r / r.norm(dim=1, keepdim=True).clamp_min(1e-12)).reshape(push[idx].shape)
            v = v + (parts["drep"] * delta.reshape(N, -1).norm(dim=1).reshape(N, 1, 1, 1) * push).to(v.dtype)
        # Same layout as forward_with_cfg: the guided velocity for both halves on the guided channels, each half's own rest.
        return torch.cat([v, torch.cat([eps, unc[:, GUIDED_CHANNELS:]], 1)], 0)

    x0 = torch.cat([z, z], 0).to(device)
    handle = model.blocks[int(parts.get("frepblock", 12))].register_forward_hook(feature_repulsion) if parts.get("frep") else None
    try:
        final = sample_fn(x0, guided, y=torch.cat([y, y]), cfg_scale=w)[-1][:N].float().cpu().numpy()
    finally:
        if handle:
            handle.remove()
    assert np.isfinite(final).all(), f"non-finite final latents: {condition}"
    return z.numpy().reshape(S, n_cls, *z.shape[1:]), final.reshape(S, n_cls, *final.shape[1:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--conditions", nargs="+", required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    torch.set_grad_enabled(False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    saved = np.load(a.tables)
    assert not set(a.seeds) & set(saved["seeds"].tolist()), "seeds overlap the calibration seeds of the tables"
    manifest = dict(script="joint_sample.py", weights=a.weights, weights_sha256=sha256(a.weights), tables=a.tables,
                    tables_sha256=sha256(a.tables), w_default=W, early_end=EARLY_END, chunk=CHUNK, grid_points=a.steps,
                    sampler="euler", guided_channels=GUIDED_CHANNELS, classes=LABELS, device=a.device)
    path = out / "joint_manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "settings differ from the existing joint manifest"
    else:
        path.write_text(json.dumps(manifest, indent=2))
    model = load_model("SiT-L/2", a.weights, a.device)
    sample_fn = Sampler(create_transport("Linear", "velocity", None, None, None)).sample_ode(
        sampling_method="euler", num_steps=a.steps, atol=1e-6, rtol=1e-3, reverse=False)
    latent_basis = torch.from_numpy(saved["latent_basis"]).to(a.device)
    embedders = None
    if any("drep" in c for c in a.conditions):
        from guidance_embed import load_models
        vae, dino, _ = load_models(a.device)
        for m in (vae, dino[0]):
            for prm in m.parameters():
                prm.requires_grad_(False)
        embedders = (vae, dino)
    for condition in a.conditions:
        folder = out / condition
        if all((folder / f"seed_{s}.npz").exists() for s in a.seeds):
            continue
        folder.mkdir(exist_ok=True)
        start = time.perf_counter()
        z, final = run_group(model, sample_fn, condition, a.seeds, a.device, latent_basis, embedders)
        seconds = time.perf_counter() - start
        for i, s in enumerate(a.seeds):
            tmp = folder / f"seed_{s}.tmp.npz"
            np.savez(tmp, seed=s, z=z[i], final=final[i], seconds=seconds / len(a.seeds), joint=True,
                     noise_sha256=array_sha256(z[i]), final_sha256=array_sha256(final[i]))
            tmp.rename(folder / f"seed_{s}.npz")
        print(f"{condition} seeds {a.seeds[0]}-{a.seeds[-1]}: {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
