"""Guidance methods. Each gets the conditional and unconditional velocities at one step and returns the velocity to integrate.

Conventions (same as SiT): t in [0, 1], t = 0 noise, t = 1 data; w = 1 is the plain conditional model (no guidance).
The predicted clean latent on SiT's linear path is x_hat = x_t + (1 - t) * v.

A method is named by a spec string `name` or `name:key=value,key=value`, e.g. `interval:lo=0.5,hi=0.8`.
Defaults below are starting points; hyperparameters marked TUNE must be searched on SiT, and VERIFY marks details
taken from paper summaries that should be checked against the paper before reporting results.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import GUIDED_CHANNELS as G


def mix(v_c, guided):
    """Guided channels 0-2 from the method; channel 3 keeps the conditional prediction, as in SiT."""
    out = v_c.clone()
    out[:, :G] = guided[:, :G]
    return out


def cfg(v_c, v_u, w):
    return v_u + w * (v_c - v_u)


class Method:
    def start(self, model, n, seed):
        self.n, self.t = n, 0.0

    def set_time(self, t):
        self.t = t

    def guide(self, v_c, v_u, x, t):
        raise NotImplementedError

    def stop(self):
        pass


class CFG(Method):
    """Standard classifier-free guidance with constant w."""

    def __init__(self, w=4.0):
        self.w = w

    def guide(self, v_c, v_u, x, t):
        return mix(v_c, cfg(v_c, v_u, self.w))


class Interval(Method):
    """Guidance Interval (Kynkäänniemi et al. 2024): w inside [lo, hi), 1 elsewhere.

    TUNE lo/hi on SiT. Default is the paper's best DiT-XL/2 interval for FID, sigma in (0.34, 1.02], converted with
    t = 1 / (1 + sigma) (same VAE latents for DiT and SiT), giving roughly t in [0.50, 0.75).
    """

    def __init__(self, w=4.0, lo=0.50, hi=0.75):
        self.w, self.lo, self.hi = w, lo, hi

    def guide(self, v_c, v_u, x, t):
        w = self.w if self.lo <= t < self.hi else 1.0
        return mix(v_c, cfg(v_c, v_u, w))


class Linear(Method):
    """Increasing schedule (Wang et al. 2024), area-normalised so the mean w over [0, 1] equals w: w(t) = 1 + 2t(w - 1)."""

    def __init__(self, w=4.0):
        self.w = w

    def guide(self, v_c, v_u, x, t):
        return mix(v_c, cfg(v_c, v_u, 1 + 2 * t * (self.w - 1)))


class EmbeddingHook(Method):
    """Base for methods that change the class embedding (rows :n conditional, rows n: null) inside the forward pass."""

    def start(self, model, n, seed):
        super().start(model, n, seed)
        self.handle = model.y_embedder.register_forward_hook(lambda _m, _i, emb: self.edit(emb))

    def stop(self):
        self.handle.remove()

    def edit(self, emb):
        """Return a replacement embedding, or None to leave it unchanged."""
        raise NotImplementedError


class CADS(EmbeddingHook):
    """CADS (Sadat et al. 2024): annealed Gaussian noise on the class embedding early in sampling, then plain CFG.

    Paper: y_hat = sqrt(g) y + s sqrt(1 - g) n, with g = 1 for tau <= tau1, linear to 0 at tau2, 0 above; tau is
    diffusion time with tau = 1 at the noise end, so tau = 1 - t here. Then rescale to the clean embedding's mean/std
    and mix with weight psi. Defaults are the paper's ImageNet-256 (w = 5) values. VERIFY: whether fresh noise is drawn
    each step and whether the null embedding is also noised (Algorithm 1); here fresh noise per step, conditional
    rows only. TUNE s/tau for w = 4.
    """

    def __init__(self, w=4.0, tau1=0.5, tau2=0.9, s=0.15, psi=1.0):
        self.w, self.tau1, self.tau2, self.s, self.psi = w, tau1, tau2, s, psi

    def start(self, model, n, seed):
        super().start(model, n, seed)
        self.gen = torch.Generator().manual_seed(10_000 + seed)

    def gamma(self):
        tau = 1 - self.t
        if tau <= self.tau1:
            return 1.0
        if tau >= self.tau2:
            return 0.0
        return (self.tau2 - tau) / (self.tau2 - self.tau1)

    def edit(self, emb):
        g = self.gamma()
        if g >= 1:
            return None
        y = emb[: self.n]
        noise = torch.randn(y.shape, generator=self.gen).to(y)
        y_hat = g ** 0.5 * y + self.s * (1 - g) ** 0.5 * noise
        mu, sd = y.mean(1, keepdim=True), y.std(1, keepdim=True)
        rescaled = (y_hat - y_hat.mean(1, keepdim=True)) / y_hat.std(1, keepdim=True) * sd + mu
        return torch.cat([self.psi * rescaled + (1 - self.psi) * y_hat, emb[self.n:]])

    def guide(self, v_c, v_u, x, t):
        return mix(v_c, cfg(v_c, v_u, self.w))


class Condition(EmbeddingHook):
    """Ours, condition space: for t < t_end, pull the class embedding toward the null embedding.

    y' = null + a (y - null), a = 1 - lam (w - 1) / w, so lam = 1 gives w * a = 1 (w = 1 to first order).
    """

    def __init__(self, w=4.0, lam=0.5, t_end=0.3):
        self.w, self.lam, self.t_end = w, lam, t_end

    def edit(self, emb):
        if self.t >= self.t_end:
            return None
        a = 1 - self.lam * (self.w - 1) / self.w
        null = emb[self.n:]
        return torch.cat([null + a * (emb[: self.n] - null), null])

    def guide(self, v_c, v_u, x, t):
        return mix(v_c, cfg(v_c, v_u, self.w))


class APG(Method):
    """Adaptive Projected Guidance (Sadat et al. 2025), applied to the predicted clean latent x_hat.

    dD = D_c - D_u; momentum dD <- dD + beta * previous (beta < 0); clip ||dD|| to r per sample; drop the part of dD
    parallel to D_c except a fraction eta; D = D_c + (w - 1) dD. Defaults are the paper's DiT-XL/2 values (VERIFY:
    order momentum -> clip -> project is from a summary of Algorithm 1). TUNE r for SiT.
    """

    def __init__(self, w=4.0, eta=0.0, r=5.0, beta=-0.5):
        self.w, self.eta, self.r, self.beta = w, eta, r, beta

    def start(self, model, n, seed):
        super().start(model, n, seed)
        self.buf = None

    def guide(self, v_c, v_u, x, t):
        s = 1 - t
        d_c = (x + s * v_c)[:, :G]
        d = (s * (v_c - v_u))[:, :G]  # = D_c - D_u
        self.buf = d if self.buf is None else d + self.beta * self.buf
        d = self.buf
        norm = d.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
        d = d * torch.clamp(self.r / norm, max=1.0)
        coef = (d * d_c).flatten(1).sum(1) / (d_c * d_c).flatten(1).sum(1).clamp_min(1e-12)
        par = coef.view(-1, 1, 1, 1) * d_c
        d = d - par + self.eta * par
        guided = (d_c + (self.w - 1) * d - x[:, :G]) / s  # back to velocity
        return mix(v_c, guided)


def laplacian_split(z):
    """One-level Laplacian pyramid: low = upsample(downsample(z)), high = z - low. VERIFY blur kernel vs FDG Algorithm 2."""
    low = F.interpolate(F.avg_pool2d(z, 2), scale_factor=2, mode="bilinear", align_corners=False)
    return low, z - low


class FDG(Method):
    """Frequency-Decoupled Guidance (Sadat et al. 2025): separate w for low and high frequencies, at every step.

    The split is linear and x_t cancels in D_c - D_u, so applying it to velocities equals applying it to x_hat.
    ImageNet setting in the paper: w_low = 1 (no low-frequency guidance), w_high = w.
    """

    def __init__(self, w_low=1.0, w_high=4.0):
        self.w_low, self.w_high = w_low, w_high

    def guide(self, v_c, v_u, x, t):
        lc, hc = laplacian_split(v_c[:, :G])
        lu, hu = laplacian_split(v_u[:, :G])
        return mix(v_c, cfg(lc, lu, self.w_low) + cfg(hc, hu, self.w_high))


class Score(Method):
    """Ours, score space: for t < t_end, remove the excess guidance along each class's shared direction.

    v = cfg - lam (w - 1) P_c (v_c - v_u), with P_c a rank-k projector fit by fit_score.py on calibration seeds that
    must not overlap the evaluation seeds.
    """

    def __init__(self, w=4.0, lam=1.0, t_end=0.3, basis=str(Path(__file__).parent / "runs" / "score_basis.npz")):
        self.w, self.lam, self.t_end, self.basis_path = w, lam, t_end, basis

    def start(self, model, n, seed):
        super().start(model, n, seed)
        saved = np.load(self.basis_path)
        assert seed not in set(saved["seeds"].tolist()), f"seed {seed} was used to fit the score basis"
        self.basis = torch.from_numpy(saved["basis"]).to(next(model.parameters()).device)  # (classes, k, 3*32*32)
        assert len(self.basis) == n

    def guide(self, v_c, v_u, x, t):
        g = cfg(v_c, v_u, self.w)
        if t < self.t_end:
            d = (v_c - v_u)[:, :G].flatten(1)
            part = torch.einsum("nk,nkd->nd", torch.einsum("nkd,nd->nk", self.basis, d), self.basis)
            g[:, :G] -= (self.w - 1) * self.lam * part.view(-1, G, 32, 32)
        return mix(v_c, g)


METHODS = {
    "cfg_w1": (CFG, dict(w=1.0)),
    "cfg_w1.5": (CFG, dict(w=1.5)),
    "cfg_w2": (CFG, dict(w=2.0)),
    "cfg_w3": (CFG, dict(w=3.0)),
    "cfg_w4": (CFG, dict(w=4.0)),
    "interval": (Interval, {}),
    "linear": (Linear, {}),
    "cads": (CADS, {}),
    "apg": (APG, {}),
    "fdg": (FDG, {}),
    "score": (Score, {}),
    "cond": (Condition, {}),
}


def folder_name(spec):
    """Spec as a folder name: a basis path is shortened to its file stem, so no '/' ends up in the name."""
    name, _, args = spec.partition(":")
    parts = [f"basis={Path(item[6:]).stem}" if item.startswith("basis=") else item for item in filter(None, args.split(","))]
    return name + (":" + ",".join(parts) if parts else "")


def build(spec):
    """'apg:r=2.5,beta=-0.75' -> APG(r=2.5, beta=-0.75) on top of the registry defaults."""
    name, _, args = spec.partition(":")
    cls, kwargs = METHODS[name]
    kwargs = dict(kwargs)
    for item in filter(None, args.split(",")):
        key, value = item.split("=")
        kwargs[key] = value if key == "basis" else float(value)
    return cls(**kwargs)
