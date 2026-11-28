"""How much of the layout collapse would a Gaussian model of each class predict? A theoretical baseline (open direction 7).

If the conditional distribution of class c and the unconditional distribution were Gaussian, N(mu_c, Sigma_c) and
N(mu_u, Sigma_u), the optimal velocity on SiT's linear path x_t = t x_1 + (1 - t) eps would be affine in x_t:
  x_hat = mu + K_t (x_t - t mu),  K_t = t Sigma (t^2 Sigma + (1 - t)^2 I)^-1,  v = (x_hat - x_t) / (1 - t)
CFG mixes the two affine fields (guidance on the first 3 latent channels only, as in SiT), so the Euler-sampled final latent
is an exact affine function of the noise, x_1 = A eps + b, and every measure below has a closed form.

Everything is computed in the low-frequency band (radius <= 4 cycles, the layout measure's band; ~49 real Fourier modes per
channel, 196 dimensions), treated as independent of the other frequencies (exact for stationary covariances). The Gaussians are
fit to the model's own samples, so the question is: if SiT were the optimal denoiser of Gaussian fits to its own w = 1 class
distributions and its unconditional distribution, how much collapse would guidance cause?
  class c    the 48 w = 1 samples of class c (seeds 0-47); unconditional: --uncond samples (null class)
  check      mu at t = 0: for any distribution, x_hat(t = 0) = E[x_1], so the model's x_0 + f(t = 0) must match the fitted mean
Predictions vs SiT (16-seed run and the window runs, seeds 0-15, same band):
  S_delta(t)    shared proportion of Delta = v_c - v_u across seeds (population value; SiT's 16-seed value is debiased)
  V_low         scale-free spread of the final latents' low band (layout diversity), at w = 1, w = 4, early-only, late-only
  per-seed R^2  how well A eps + b predicts SiT's own final low band for the same noise (class-centered), w = 1 and w = 4
usage: gaussian_theory.py --sample-uncond --weights <file> --out <dir> --n-batches 24     (GPU: null-class samples)
       gaussian_theory.py --theory --uncond <dir>                                          -> table + figure
"""
import argparse, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "bench")
from common import LABELS, NULL_CLASS, initial_noise
from guidance_window import band_masks, spread_v

O = Path("outputs/guidance_delta")
W, EARLY_END, STEPS = 4.0, 0.3, 49
UNCOND_SEED0 = 100000  # noise seeds for the unconditional samples, far from every evaluation seed


def sample_uncond(weights, out, n_batches, device="mps"):
    import torch
    from common import load_model
    from transport import Sampler, create_transport
    torch.set_grad_enabled(False)
    model = load_model("SiT-L/2", weights, device)
    sample_fn = Sampler(create_transport("Linear", "velocity", None, None, None)).sample_ode(
        sampling_method="euler", num_steps=50, atol=1e-6, rtol=1e-3, reverse=False)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for i in range(n_batches):
        target = out / f"batch_{i}.npz"
        if target.exists():
            continue
        seeds = [UNCOND_SEED0 + 2 * i, UNCOND_SEED0 + 2 * i + 1]
        z = torch.cat([initial_noise(s) for s in seeds], 0).to(device)  # 16 rows, one forward pass per step
        y = torch.full((len(z),), NULL_CLASS, device=device)
        final = sample_fn(z, model.forward, y=y)[-1].float().cpu().numpy()
        np.savez(target, seeds=seeds, z=z.cpu().numpy(), final=final)
        print("uncond batch", i, flush=True)


CUTOFF, MEANS, SHRINK = 4, "network", "lw"  # set from the command line


def band_basis():
    """Real orthonormal basis of the low band (radius <= CUTOFF cycles) of one 32 x 32 channel: eigenvectors of the (symmetric)
    low-pass operator."""
    f = np.fft.fftfreq(32, 1 / 32)
    low = np.hypot(*np.meshgrid(f, f, indexing="ij")) <= CUTOFF
    eye = np.eye(1024).reshape(1024, 32, 32)
    proj = np.fft.ifft2(np.fft.fft2(eye) * low).real.reshape(1024, 1024)
    lam, vec = np.linalg.eigh((proj + proj.T) / 2)
    return vec[:, lam > 0.5]  # (1024, d1)


def coords(latents, Q):
    """latents (..., 4, 32, 32) -> (..., 4 * d1) band coordinates, channel-major."""
    x = latents.reshape(*latents.shape[:-2], 1024)
    return np.concatenate([x[..., c, :] @ Q for c in range(4)], -1)


def shrunk_cov(X, mean=None):
    """Shrinkage of the sample covariance (around `mean` if given) towards a scaled identity: Ledoit-Wolf, or a fixed weight."""
    n, d = X.shape
    Xc = X - (X.mean(0) if mean is None else mean)
    S = Xc.T @ Xc / n
    mu = np.trace(S) / d
    b2 = min(sum(((np.outer(x, x) - S) ** 2).sum() for x in Xc) / n ** 2, ((S - mu * np.eye(d)) ** 2).sum())
    a = b2 / ((S - mu * np.eye(d)) ** 2).sum() if SHRINK == "lw" else float(SHRINK)
    return a * mu * np.eye(d) + (1 - a) * S


def field(mu, Sig, t):
    """Affine optimal velocity v = F x + g at time t for x_1 ~ N(mu, Sig)."""
    d = len(mu)
    K = t * Sig @ np.linalg.inv(t * t * Sig + (1 - t) ** 2 * np.eye(d))
    return (K - np.eye(d)) / (1 - t), (np.eye(d) - t * K) @ mu / (1 - t)


def flow(mu_c, S_c, mu_u, S_u, G, wfun):
    """Euler on t_k = k / 49 with CFG scale wfun(t). Returns A, b (x_1 = A eps + b) and per-step (F_delta, g_delta, A_t, b_t)."""
    d = len(mu_c)
    A, b, hist = np.eye(d), np.zeros(d), []
    for k in range(STEPS):
        t, dt = k / STEPS, 1 / STEPS
        Fc, gc = field(mu_c, S_c, t); Fu, gu = field(mu_u, S_u, t)
        w = wfun(t)
        F = G @ (Fu + w * (Fc - Fu)) + (np.eye(d) - G) @ Fc  # guided channels mix; channel 3 takes the conditional field
        g = G @ (gu + w * (gc - gu)) + (np.eye(d) - G) @ gc
        hist.append((Fc - Fu, gc - gu, A.copy(), b.copy()))
        A, b = A + dt * F @ A, b + dt * (F @ b + g)
    return A, b, hist


def v_low_gauss(A, b, mean_sq_extra=0.0):
    """Population scale-free spread tr(C) / E|x|^2 of x = A eps + b."""
    tr = np.trace(A @ A.T)
    return tr / (tr + b @ b)


def theory(uncond):
    Q = band_basis(); d1 = Q.shape[1]
    G = np.diag(np.repeat([1.0, 1.0, 1.0, 0.0], d1))
    # Class samples at w = 1: seeds 0-15 from the 16-seed run, 16-47 from SiT-L_2_night.
    w1 = np.concatenate([np.stack([np.load(O / "SiT-L_2_16seeds/w1" / f"seed_{s}/trajectory.npz")["final"] for s in range(16)]),
                         np.stack([np.load(O / "SiT-L_2_night/w1" / f"seed_{s}.npz")["final"] for s in range(16, 48)])]).astype(np.float64)
    U = np.concatenate([np.load(p)["final"] for p in sorted(Path(uncond).glob("batch_*.npz"))]).astype(np.float64)
    Cw1, Cu = coords(w1, Q), coords(U, Q)  # (48, 8, D), (n_u, D)
    mu_u = Cu.mean(0)
    S_u = None
    traj = {w: [np.load(O / f"SiT-L_2_16seeds/{w}" / f"seed_{s}/trajectory.npz") for s in range(16)] for w in ("w1", "w4")}
    # check: the network's own E[x_1] at t = 0, x_0 + f(t = 0), averaged over seeds, against the fitted means
    f0c = coords(np.stack([t["x_t"][0] + t["f_cond"][0] for t in traj["w1"]]).astype(np.float64), Q).mean(0)  # (8, D)
    f0u = coords(np.stack([t["x_t"][0] + t["f_uncond"][0] for t in traj["w1"]]).astype(np.float64), Q).mean((0, 1))
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
    if MEANS == "network":
        # E[x_1] = x_0 + f(t = 0) holds for any distribution; the network's value (averaged over 16 seeds) is far less noisy than
        # a 48-sample mean when the spread is large. Covariances are then taken around these means.
        mu_u_net, mu_c_net = f0u, f0c
    S_u = shrunk_cov(Cu, mu_u_net if MEANS == "network" else None)
    print(f"{len(U)} unconditional samples; band radius <= {CUTOFF}, dimension {4 * d1}; means: {MEANS}; shrinkage: {SHRINK}")
    print(f"check E[x_1]: network x_0 + f_uncond(0) vs uncond sample mean: rel. diff {rel(f0u, mu_u):.3f}; "
          f"x_0 + f_cond(0) vs w = 1 class means: rel. diff {np.mean([rel(f0c[i], Cw1[:, i].mean(0)) for i in range(8)]):.3f}")
    finals = {w: coords(np.stack([t["final"] for t in traj[w]]).astype(np.float64), Q) for w in ("w1", "w4")}
    for name, run in (("early", "SiT-L_2_window_early"), ("late", "SiT-L_2_window_late")):
        finals[name] = coords(np.stack([np.load(O / run / "w4" / f"seed_{s}/trajectory.npz")["final"] for s in range(16)]).astype(np.float64), Q)
    eps = coords(np.stack([t["x_t"][0] for t in traj["w1"]]).astype(np.float64), Q)  # (16, 8, D) noise in band coordinates
    sched = dict(w1=lambda t: 1.0, w4=lambda t: W, early=lambda t: W if t < EARLY_END else 1.0, late=lambda t: 1.0 if t < EARLY_END else W)
    rows, S_theory = [], []
    for i in range(8):
        mu_c = mu_c_net[i] if MEANS == "network" else Cw1[:, i].mean(0)
        S_c = shrunk_cov(Cw1[:, i], mu_c if MEANS == "network" else None)
        res = {k: flow(mu_c, S_c, mu_u_net if MEANS == "network" else mu_u, S_u, G, f) for k, f in sched.items()}
        row = dict(cls=i)
        for k, (A, b, _) in res.items():
            row[f"V_gauss_{k}"] = v_low_gauss(A, b)
            row[f"V_sit_{k}"] = float(spread_v(finals[k][:, i]))
        for k in ("w1", "w4"):
            A, b, _ = res[k]
            pred, act = eps[:, i] @ A.T + b, finals[k][:, i]
            pc, ac = pred - pred.mean(0), act - act.mean(0)
            row[f"R2_seed_{k}"] = 1 - ((ac - pc) ** 2).sum() / (ac ** 2).sum()
            row[f"cos_seed_{k}"] = float(np.mean([(p @ a) / np.linalg.norm(p) / np.linalg.norm(a) for p, a in zip(pc, ac)]))
        S_t = []
        for Fd, gd, At, bt in res["w4"][2]:
            mD = Fd @ bt + gd
            S_t.append((mD @ mD) / (mD @ mD + np.trace(Fd @ At @ At.T @ Fd.T)))
        S_theory.append(S_t)
        rows.append(row)
    S_theory = np.array(S_theory).mean(0)
    # SiT's S(Delta) in the same band, w = 4, debiased for 16 seeds: (N S - 1) / (N - 1).
    D = np.stack([coords(t["delta"].astype(np.float64), Q) for t in traj["w4"]])  # (16, 49, 8, D)
    S_hat = ((D.mean(0) ** 2).sum(-1) / (D ** 2).sum(-1).mean(0))  # (49, 8)
    S_sit = ((16 * S_hat - 1) / 15).mean(1)
    print("\nS(Delta) in the low band, w = 4 (theory: population; SiT: 16 seeds, debiased)")
    for k in (0, 3, 6, 10, 15, 20, 25, 35, 48):
        print(f"  s_t {k / 49:.2f}: Gaussian {S_theory[k]:.3f}   SiT {S_sit[k]:.3f}")
    mean = lambda key: np.mean([r[key] for r in rows])
    print("\nlayout spread V_low (mean over classes)      Gaussian    SiT")
    for k in ("w1", "w4", "early", "late"):
        print(f"  {k:6}                                      {mean(f'V_gauss_{k}'):.3f}     {mean(f'V_sit_{k}'):.3f}")
    for src in ("gauss", "sit"):
        r = lambda k: np.mean([x[f"V_{src}_{k}"] / x[f"V_{src}_w1"] for x in rows])
        sh = lambda k: np.mean([(x[f"V_{src}_{k}"] - x[f"V_{src}_w1"]) / (x[f"V_{src}_w4"] - x[f"V_{src}_w1"]) for x in rows])
        print(f"  {src}: V_w4 / V_w1 = {r('w4'):.2f}; share of the w = 4 layout loss reproduced: early-only {sh('early'):.2f}, late-only {sh('late'):.2f}")
    print("\nper-seed: Gaussian A eps + b vs SiT's own final low band, same noise (class-centered; mean over classes)")
    for k in ("w1", "w4"):
        print(f"  {k}: R^2 {mean(f'R2_seed_{k}'):.2f}, mean cosine {mean(f'cos_seed_{k}'):.2f}   per class R^2 " + ", ".join(f"{r[f'R2_seed_{k}']:.2f}" for r in rows))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    t = np.arange(STEPS) / STEPS
    ax.plot(t, S_theory, color="#2a6fdb", label="Gaussian model"); ax.plot(t, S_sit, color="#d9480f", label="SiT-L/2")
    ax.axvline(EARLY_END, color="#bbb", lw=0.8); ax.set(xlabel="s_t", ylabel="shared proportion S(Δ), low band", title="Shared guidance, w = 4")
    ax.legend(); fig.tight_layout(); fig.savefig(O / "figures" / f"gaussian_shared_r{CUTOFF}_{MEANS}_{SHRINK}.png", dpi=150)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample-uncond", action="store_true"); p.add_argument("--theory", action="store_true")
    p.add_argument("--weights"); p.add_argument("--out"); p.add_argument("--uncond"); p.add_argument("--n-batches", type=int, default=24)
    p.add_argument("--cutoff", type=int, default=4); p.add_argument("--means", choices=["network", "sample"], default="network")
    p.add_argument("--shrink", default="lw", help="'lw' (Ledoit-Wolf) or a fixed weight in [0, 1]")
    a = p.parse_args()
    CUTOFF, MEANS, SHRINK = a.cutoff, a.means, a.shrink
    if a.sample_uncond:
        sample_uncond(a.weights, a.out, a.n_batches)
    if a.theory:
        theory(a.uncond)
