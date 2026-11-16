"""Shared pieces: classes, noise, model loading, and one Euler sampler that every guidance method plugs into.

Sampling matches SiT's own `Sampler.sample_ode(sampling_method="euler")` with `model.forward_with_cfg`: a fixed grid of
STEPS points on t in [0, 1] (t = 0 is noise, t = 1 is data), so STEPS - 1 model calls, and guidance on the first
GUIDED_CHANNELS latent channels only. `check.py` verifies this equivalence.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sit"))
from models import SiT_models  # noqa: E402  (the unmodified upstream SiT code)

LABELS = [207, 360, 387, 974, 88, 979, 417, 279]
LABEL_NAMES = ["golden retriever", "otter", "lesser panda", "geyser", "macaw", "valley", "balloon", "arctic fox"]
NULL_CLASS = 1000
GUIDED_CHANNELS = 3  # SiT's forward_with_cfg guides channels 0-2; channel 3 always takes the conditional prediction
STEPS = 50  # grid points; Euler makes STEPS - 1 = 49 model calls
N_RECORD = 25  # steps at which x_hat is saved (every other step; dense enough to resolve the early t < 0.3 window)
DEFAULT_WEIGHTS = ROOT / "pretrained_models" / "SiT-XL-2-256x256.pt"


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(weights=DEFAULT_WEIGHTS, name="SiT-XL/2", device=None):
    """weights=None gives a randomly initialised model (used by check.py)."""
    model = SiT_models[name](input_size=32, num_classes=1000, learn_sigma=True)
    if weights is not None:
        state = torch.load(weights, map_location="cpu", weights_only=True)  # weights_only: never unpickle code
        model.load_state_dict(state.get("ema", state), strict=True)
    # eval() matters: in train mode the label embedder randomly drops labels to the null class.
    return model.to(device or default_device()).eval()


def initial_noise(seed, n=len(LABELS)):
    """Noise depends only on the seed (CPU generator), so every method starts from identical noise."""
    return torch.randn(n, 4, 32, 32, generator=torch.Generator().manual_seed(seed))


def record_steps(calls):
    return sorted(set(np.linspace(0, calls - 1, N_RECORD).round().astype(int).tolist()))


@torch.no_grad()
def sample(model, method, seed, labels=LABELS, steps=STEPS, stop_at=None):
    """Sample one batch (one image per label) with a guidance method.

    Returns final latents (n, 4, 32, 32), x_hat at N_RECORD steps (k, n, 4, 32, 32) and their times.
    stop_at: end early once t >= stop_at (fit_score.py only needs the early steps).
    """
    device = next(model.parameters()).device
    n = len(labels)
    x = initial_noise(seed, n).to(device)
    y = torch.tensor(list(labels) + [NULL_CLASS] * n, device=device)
    grid = torch.linspace(0, 1, steps).tolist()
    keep = record_steps(steps - 1)
    xhat, times = [], []
    method.start(model, n, seed)
    try:
        for k in range(steps - 1):
            t = grid[k]
            if stop_at is not None and t >= stop_at:
                break
            method.set_time(t)  # hooks inside the forward pass (CADS, condition) read the current time
            # model(...) goes through nn.Module.__call__, so forward hooks on model.y_embedder fire.
            out = model(torch.cat([x, x]), torch.full((2 * n,), t, device=device), y).float()
            v = method.guide(out[:n], out[n:], x, t)
            if k in keep:
                xhat.append((x + (1 - t) * v).cpu())  # predicted clean latent on the linear path
                times.append(t)
            x = x + (grid[k + 1] - t) * v
    finally:
        method.stop()
    return dict(final=x.cpu().numpy(), xhat=torch.stack(xhat).numpy().astype(np.float16) if xhat else None,
                t=np.array(times))


def parse_seeds(items):
    """['100-103', '7'] -> [100, 101, 102, 103, 7]"""
    seeds = []
    for item in items:
        lo, _, hi = item.partition("-")
        seeds += list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    return seeds
