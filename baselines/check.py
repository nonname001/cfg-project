"""Correctness checks that need no weights: a small randomly initialised SiT on CPU.

1. common.sample with plain CFG reproduces SiT's own Sampler.sample_ode(euler) + forward_with_cfg.
2. Every method reduces exactly to plain CFG at its neutral setting (e.g. APG with eta=1, r=inf, beta=0).
usage: python check.py      (needs torchdiffeq for check 1)
"""
import tempfile
from pathlib import Path

import numpy as np
import torch

import methods
from common import LABELS, NULL_CLASS, STEPS, initial_noise, load_model, sample


def tiny_model():
    torch.manual_seed(0)
    model = load_model(weights=None, name="SiT-S/2", device="cpu")
    with torch.no_grad():  # SiT zero-initialises its output layers; perturb so outputs are not trivially zero
        for p in model.parameters():
            p.add_(0.02 * torch.randn_like(p))
    return model


def check_against_sit(model, w=4.0, seed=3):
    from transport import Sampler, create_transport
    n = len(LABELS)
    z = initial_noise(seed)
    y = torch.tensor(LABELS + [NULL_CLASS] * n)
    sample_fn = Sampler(create_transport("Linear", "velocity", None, None, None)).sample_ode(
        sampling_method="euler", num_steps=STEPS, atol=1e-6, rtol=1e-3, reverse=False)
    with torch.no_grad():
        ref = sample_fn(torch.cat([z, z]), model.forward_with_cfg, y=y, cfg_scale=w)[-1][:n].numpy()
    ours = sample(model, methods.build(f"cfg_w{w:g}"), seed)["final"]
    err = np.abs(ours - ref).max() / np.abs(ref).max()
    print(f"[1] own sampler vs SiT sample_ode + forward_with_cfg: max relative error {err:.2e}")
    assert err < 1e-5


def check_neutral(model, seed=3):
    with tempfile.TemporaryDirectory() as tmp:
        basis = Path(tmp) / "basis.npz"
        rng = np.random.default_rng(0)
        np.savez(basis, basis=rng.standard_normal((len(LABELS), 3, 3 * 32 * 32)).astype(np.float32), seeds=np.array([0]))
        ref = sample(model, methods.build("cfg_w4"), seed)["final"]
        neutral = ["interval:lo=0,hi=2", "linear:w=1", "cads:tau1=1.5,tau2=2", "apg:eta=1,r=1e9,beta=0",
                   "fdg:w_low=4,w_high=4", f"score:lam=0,basis={basis}", "cond:lam=0"]
        for spec in neutral:
            target = sample(model, methods.build("cfg_w1"), seed)["final"] if spec == "linear:w=1" else ref
            out = sample(model, methods.build(spec), seed)["final"]
            err = np.abs(out - target).max() / np.abs(target).max()
            print(f"[2] {spec:40s} vs plain CFG: max relative error {err:.2e}")
            assert err < 1e-5, spec
        # and the active settings must actually change the samples
        for spec in ["interval", "cads", "apg", "fdg", f"score:basis={basis}", "cond"]:
            diff = np.abs(sample(model, methods.build(spec), seed)["final"] - ref).max()
            print(f"[3] {spec:40s} differs from plain CFG by {diff:.3g}")
            assert diff > 1e-4, spec


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    m = tiny_model()
    try:
        check_against_sit(m)
    except ModuleNotFoundError as e:
        print(f"[1] skipped: {e}")
    check_neutral(m)
    print("all checks passed")
