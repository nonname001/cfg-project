"""Damp the shared class component of guidance during early steps, in four spaces, at one compute budget.

Every condition runs the same Euler grid and 2 model passes per step, so all use 98 branch evaluations. w = 4 throughout except
where a condition changes guidance during s_t < EARLY_END. lambda is the damping strength; lambda = 1 aims to remove all excess
guidance along the targeted component, that is, guidance beyond w = 1. With excess guidance (w-1)*Delta, the methods are:

  scalar_L   w_early = 1 + (w-1)(1-L). Damps every direction equally: the control for whether targeting matters.
  score_L    f = f_cond + (w-1)(Delta - L P_c Delta), P_c = rank-3 projector onto class c's shared guidance subspace.
             The linear path makes score and velocity differ by a scalar per step, so this projection is the same in both.
  freq_L     as score_L, with P replaced by a low-pass filter at LOW_CUTOFF cycles. No calibration needed.
  cond_L     class embedding y' = null + a(y - null), a = 1 - L(w-1)/w, so that w*a = 1 at L=1 to first order.
  feat_L     in blocks 13-23, conditional rows lose (w-1)/w * L of the pooled cond-uncond difference inside a per-class rank-3
             subspace, again targeting w*(1 - (w-1)/w) = 1 to first order.
  wrongclass_L  score_L using the next class's subspace: tests whether class-specific directions matter.

score, freq and scalar act on the 3 guided channels only; cond and feat change f_cond, so they also reach channel 3.

Subspaces are fit on the calibration seeds of a guidance_capture.py run (--build-tables); evaluate on other seeds.
usage: guidance_intervene.py --build-tables --calibration <run> --tables <file>
       guidance_intervene.py --tables <file> --weights <file> --out <dir> --seeds 16 17 ... [--conditions w1 w4 ...]
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "bench")
from guidance_delta import GUIDED_CHANNELS, LABELS, NULL_CLASS, array_sha256, initial_noise, load_model, sha256
from transport import Sampler, create_transport

W = 4.0
EARLY_END = 0.3
RANK = 3
LOW_CUTOFF = 4
FEATURE_BLOCKS = list(range(13, 24))
CONDITIONS = ["w1", "w4", "scalar_0.5", "scalar_1", "score_0.5", "score_1", "freq_0.5", "freq_1", "cond_0.5", "cond_1",
              "feat_0.5", "feat_1", "wrongclass_1"]


def build_tables(run, path, seeds=range(16)):
    """Rank-3 bases per class from the mean Delta (latent) and mean cond-uncond difference (features) over early steps.

    Means are taken over seeds, so seed-specific parts cancel and what remains is the shared component. An intervened trajectory lies
    between w=1 and w=4, and the shared direction differs between them later in the window, so both are included.
    """
    run = Path(run)
    latents, features = [], []
    for w in ("w1", "w4"):
        files = [np.load(run / w / f"seed_{s}" / "trajectory.npz") for s in seeds]
        s_t = files[0]["s_t"]
        early = s_t < EARLY_END
        delta = np.stack([f["delta"][early][:, :, :GUIDED_CHANNELS] for f in files], 0).astype(np.float64)
        latents.append(delta.mean(0).reshape(early.sum(), 8, -1))  # (steps, classes, 3072)
        pooled = [np.load(run / w / f"seed_{s}" / "pooled.npy", mmap_mode="r") for s in seeds]
        diff = np.mean([np.asarray(p[early][:, FEATURE_BLOCKS, 0] - p[early][:, FEATURE_BLOCKS, 1], dtype=np.float64) for p in pooled], 0)
        features.append(diff)  # (steps, blocks, classes, 1024)
    latents, features = np.concatenate(latents, 0), np.concatenate(features, 0)
    top = lambda x: np.linalg.svd(x, full_matrices=False)[2][:RANK]
    latent_basis = np.stack([top(latents[:, c]) for c in range(8)]).astype(np.float32)  # (8, 3, 3072), orthonormal rows
    feature_basis = np.stack([np.stack([top(features[:, b, c]) for c in range(8)]) for b in range(len(FEATURE_BLOCKS))]).astype(np.float32)
    np.savez(path, latent_basis=latent_basis, feature_basis=feature_basis, seeds=np.array(list(seeds)), early_end=EARLY_END, rank=RANK)


def low_pass_matrix():
    """Real 1024 x 1024 matrix for the radial low-pass filter, so it can run as one matmul on any device."""
    f = np.fft.fftfreq(32, 1 / 32)
    mask = np.hypot(*np.meshgrid(f, f, indexing="ij")) <= LOW_CUTOFF
    eye = np.eye(1024).reshape(1024, 32, 32)
    return np.fft.ifft2(np.fft.fft2(eye) * mask).real.reshape(1024, 1024).astype(np.float32)


def parse(condition):
    if condition in ("w1", "w4"):
        return condition, 0.0
    method, strength = condition.split("_")
    return method, float(strength)


class Intervention:
    """score and cond interventions in a form guidance_capture.py can drive. A spec joins parts with '+', e.g. 'score_1+cond_0.5'.

    The arithmetic matches run_condition exactly, so a single-part spec reproduces its samples bit for bit. In a mix, cond changes
    the embedding first, so score projects the Delta that the damped embedding produced.
    """

    def __init__(self, spec, tables, n):
        self.parts = dict(parse(part) for part in spec.split("+"))
        assert set(self.parts) <= {"score", "cond"}, f"unsupported intervention: {spec}"
        self.tables, self.n, self.active, self.handles = tables, n, False, []

    def set_time(self, t):
        self.active = t < EARLY_END

    def install(self, model):
        if "cond" in self.parts:
            alpha = 1 - self.parts["cond"] * (W - 1) / W
            n = self.n

            def embed_hook(_module, _inputs, emb):
                if not self.active:
                    return None
                return torch.cat([emb[n:] + alpha * (emb[:n] - emb[n:]), emb[n:]], 0)
            self.handles.append(model.y_embedder.register_forward_hook(embed_hook))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def adjust(self, out, last):
        """out: forward_with_cfg's result; last: the raw branch outputs it was built from. Returns (out, removed fraction)."""
        if not (self.active and "score" in self.parts):
            return out, 0.0
        n = self.n
        delta = (last[:n, :GUIDED_CHANNELS] - last[n:, :GUIDED_CHANNELS]).float()
        basis = self.tables["latent"]
        flat = delta.reshape(n, -1)
        part = torch.einsum("nk,nkd->nd", torch.einsum("nkd,nd->nk", basis, flat), basis).reshape(delta.shape)
        out = out.clone()
        out[:n, :GUIDED_CHANNELS] -= (W - 1) * self.parts["score"] * part
        return out, float((part ** 2).sum() / (delta ** 2).sum())


def load_tables(path, device):
    saved = np.load(path)
    return dict(latent=torch.from_numpy(saved["latent_basis"]).to(device), feature=torch.from_numpy(saved["feature_basis"]).to(device),
                low_pass=torch.from_numpy(low_pass_matrix()).to(device)), saved["seeds"].tolist()


def run_condition(model, sample_fn, condition, seed, device, tables):
    """Sample one batch. Returns final latents and, per step, the fraction of ||Delta||^2 the condition removed from guidance."""
    method, strength = parse(condition)
    n = len(LABELS)
    z = initial_noise(seed)
    x = torch.cat([z, z], 0).to(device)
    y = torch.tensor(LABELS + [NULL_CLASS] * n, device=device)
    state = dict(active=False, last=None)
    removed = []
    excess = (W - 1) / W

    basis = None
    if method in ("score", "wrongclass"):
        basis = tables["latent"] if method == "score" else tables["latent"].roll(-1, 0)

    forward = model.forward

    def recording_forward(xb, tb, yb):
        state["last"] = forward(xb, tb, yb)
        return state["last"]

    def guided(xb, tb, **kwargs):
        state["active"] = float(tb[0]) < EARLY_END
        if method == "w1" or (method == "scalar" and state["active"]):
            kwargs["cfg_scale"] = 1.0 if method == "w1" else 1 + (W - 1) * (1 - strength)
        out = model.forward_with_cfg(xb, tb, **kwargs)
        fraction = 0.0
        if state["active"] and method in ("score", "wrongclass", "freq"):
            last = state["last"]
            delta = (last[:n, :GUIDED_CHANNELS] - last[n:, :GUIDED_CHANNELS]).float()
            if method == "freq":
                part = (delta.reshape(n, GUIDED_CHANNELS, 1024) @ tables["low_pass"].T).reshape(delta.shape)
            else:
                flat = delta.reshape(n, -1)
                part = torch.einsum("nk,nkd->nd", torch.einsum("nkd,nd->nk", basis, flat), basis).reshape(delta.shape)
            out = out.clone()
            out[:n, :GUIDED_CHANNELS] -= (W - 1) * strength * part  # excess guidance loses L * part
            fraction = float((part ** 2).sum() / (delta ** 2).sum())
        removed.append(fraction)
        return out

    def embed_hook(_module, _inputs, emb):  # rows [:n] are class embeddings, rows [n:] the null embedding
        if not state["active"]:
            return None
        alpha = 1 - strength * excess
        return torch.cat([emb[n:] + alpha * (emb[:n] - emb[n:]), emb[n:]], 0)

    def block_hook(index):
        def hook(_module, _inputs, out):
            if not state["active"]:
                return None
            u = tables["feature"][index]  # (8, 3, 1024)
            diff = out[:n].mean(1) - out[n:].mean(1)
            part = torch.einsum("nk,nkd->nd", torch.einsum("nkd,nd->nk", u, diff), u)
            return torch.cat([out[:n] - excess * strength * part[:, None, :], out[n:]], 0)  # same shift on every token
        return hook

    handles = []
    if method == "cond":
        handles.append(model.y_embedder.register_forward_hook(embed_hook))
    if method == "feat":
        handles += [model.blocks[b].register_forward_hook(block_hook(i)) for i, b in enumerate(FEATURE_BLOCKS)]
    model.forward = recording_forward
    try:
        # Hooks read state["active"], which guided sets before the model runs.
        final = sample_fn(x, guided, y=y, cfg_scale=W)[-1][:n].float().cpu().numpy()
    finally:
        del model.__dict__["forward"]
        for handle in handles:
            handle.remove()
    assert np.isfinite(final).all(), f"non-finite final latents: {condition} seed {seed}"
    return z.numpy(), final, np.array(removed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build-tables", action="store_true")
    p.add_argument("--calibration")
    p.add_argument("--tables", required=True)
    p.add_argument("--weights")
    p.add_argument("--out")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    if a.build_tables:
        build_tables(a.calibration, a.tables)
        return

    torch.set_grad_enabled(False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    saved = np.load(a.tables)
    assert not set(a.seeds) & set(saved["seeds"].tolist()) or "calibration" in out.name, "evaluation seeds overlap the calibration seeds"
    tables = dict(latent=torch.from_numpy(saved["latent_basis"]).to(a.device), feature=torch.from_numpy(saved["feature_basis"]).to(a.device),
                  low_pass=torch.from_numpy(low_pass_matrix()).to(a.device))
    manifest = dict(weights=a.weights, weights_sha256=sha256(a.weights), tables=a.tables, tables_sha256=sha256(a.tables), w=W, early_end=EARLY_END,
                    rank=RANK, low_cutoff=LOW_CUTOFF, feature_blocks=FEATURE_BLOCKS, grid_points=a.steps, model_calls=a.steps - 1,
                    branch_evaluations=2 * (a.steps - 1), sampler="euler", guided_channels=GUIDED_CHANNELS, classes=LABELS, device=a.device,
                    calibration_seeds=saved["seeds"].tolist())
    path = out / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "settings differ from the existing manifest in --out"
    else:
        path.write_text(json.dumps(manifest, indent=2))

    model = load_model("SiT-L/2", a.weights, a.device)
    sample_fn = Sampler(create_transport("Linear", "velocity", None, None, None)).sample_ode(
        sampling_method="euler", num_steps=a.steps, atol=1e-6, rtol=1e-3, reverse=False)
    for seed in a.seeds:  # seeds are the outer loop, so stopping early leaves every condition on the same seeds
        for condition in a.conditions:
            target = out / condition / f"seed_{seed}.npz"
            if target.exists():
                continue
            target.parent.mkdir(exist_ok=True)
            start = time.perf_counter()
            z, final, removed = run_condition(model, sample_fn, condition, seed, a.device, tables)
            seconds = time.perf_counter() - start
            tmp = target.with_suffix(".tmp.npz")
            np.savez(tmp, seed=seed, z=z, final=final, removed=removed, seconds=seconds, noise_sha256=array_sha256(z), final_sha256=array_sha256(final))
            tmp.rename(target)  # a file at the final path is always complete
            print(f"{condition} seed={seed}: {seconds:.1f}s removed(t=0)={removed[0]:.3f}", flush=True)


if __name__ == "__main__":
    main()
