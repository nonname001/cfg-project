"""Log the CFG guidance vector along real sampling trajectories, in the model's native velocity parameterization.

    Delta_t = v_cond(x_t, t) - v_uncond(x_t, t)      # all 4 latent channels, 4096 dims per sample

Both branches see the same x_t (the guided trajectory). SiT time runs t=0 (noise) -> t=1 (data). Sampling goes through the
unmodified repo path, Sampler.sample_ode + model.forward_with_cfg; the only intervention is an instance-level wrapper on
model.forward (forward_with_cfg calls self.forward directly, which bypasses nn.Module hooks).

A fixed-grid Euler solver is required: an adaptive solver has no fixed t grid and evaluates rejected steps, so
Delta_{t-1} would not be well defined.

usage: guidance_delta.py --weights <file> --out <dir> [--model SiT-L/2] [--seeds 0 1 ...] [--steps 50] [--cfg 4.0]
Run from the project root. Writes one seed_<n>.npz per seed plus manifest.json.
"""
import argparse, hashlib, json, platform, subprocess, sys, time
import importlib.metadata as md
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "sit")
from models import SiT_models
from transport import Sampler, create_transport

LABELS = [207, 360, 387, 974, 88, 979, 417, 279]  # sample.py's default class_labels
LABEL_NAMES = ["golden retriever", "otter", "lesser panda", "geyser", "macaw", "valley", "balloon", "arctic fox"]
NULL_CLASS = 1000
GUIDED_CHANNELS = 3  # forward_with_cfg guides model_out[:, :3] only


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_model(name, weights, device):
    # input_size is in latent pixels: 256px images / 8 (VAE downsampling) = 32. learn_sigma=True matches the 256x256
    # checkpoints, which have 8 output channels; SiT.forward() drops the 4 sigma channels before returning.
    model = SiT_models[name](input_size=32, num_classes=1000, learn_sigma=True)
    # Neither branch can run code from the file: safetensors holds only tensors, and weights_only=True refuses pickled objects.
    if weights.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(weights)
    else:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    # strict=True fails on any missing or extra key, so weights for the wrong architecture cannot load silently.
    model.load_state_dict(state, strict=True)
    # eval() matters: in train mode LabelEmbedder randomly replaces labels with the null class.
    return model.to(device).eval()


def initial_noise(seed):
    """CPU generator, so the noise does not depend on the device. Matches bench/prep.py for the same seed."""
    # A private generator keeps the noise independent of global RNG state. This is not the noise sample.py makes for the same
    # --seed: sample.py draws from the global RNG after model init has already consumed part of it.
    return torch.randn(len(LABELS), 4, 32, 32, generator=torch.Generator().manual_seed(seed))


def run_seed(model, sample_fn, seed, cfg, device, record=True):
    """Sample one batch. Returns final latents, plus the per-step log when record=True."""
    n = len(LABELS)
    z = initial_noise(seed)
    # CFG batch layout: rows [:n] are conditional and rows [n:] are unconditional, with the null class as their label.
    x = torch.cat([z, z], 0).to(device)
    y = torch.tensor(LABELS + [NULL_CLASS] * n, device=device)
    log = dict(t=[], x=[], cond=[], uncond=[])
    # Keep the bound method: recording_forward must call the real forward, not itself.
    forward = model.forward

    # forward_with_cfg passes cat([half, half]), so both branches see the same x_t and out[:n] - out[n:] is exactly
    # v_cond - v_uncond at one point. Every tb entry is the same scalar time. The raw outputs are logged, before guidance mixes them.
    def recording_forward(xb, tb, yb):
        out = forward(xb, tb, yb)
        log["t"].append(float(tb[0]))
        log["x"].append(xb[:n].float().cpu())
        log["cond"].append(out[:n].float().cpu())
        log["uncond"].append(out[n:].float().cpu())
        return out

    if record:
        # An instance attribute shadows the class method. forward_with_cfg calls self.forward(...) directly, so nn.Module forward
        # hooks would never fire. Shadowing it captures both branches without editing sit/models.py.
        model.forward = recording_forward
    try:
        # sample_fn returns the state at every grid point, shape (steps, 2n, 4, 32, 32). [-1] takes the last grid point and [:n]
        # the conditional rows, which are the real samples. The rows [n:] are discarded, as in sample.py: their 4th channel drifts
        # from the conditional rows because forward_with_cfg leaves it unguided, but forward_with_cfg only ever reads x[:n].
        final = sample_fn(x, model.forward_with_cfg, y=y, cfg_scale=cfg)[-1][:n].float().cpu().numpy()
    finally:
        if record:
            # Removing the shadow restores the class method, even if sampling raised.
            del model.__dict__["forward"]
    return z.numpy(), final, log


def metrics(delta):
    """delta: (T, B, C, H, W). L2 norm per step and cosine between consecutive steps, per sample, in float64."""
    # Flatten each sample to one vector (4096 dims for 4 channels). float64 keeps rounding out of the 4096-term dot products.
    d = delta.reshape(delta.shape[0], delta.shape[1], -1).astype(np.float64)
    norm = np.linalg.norm(d, axis=-1)
    # cos has T-1 rows: cos[k] compares step k+1 with step k, so it belongs to time t[k+1]. Step 0 has no predecessor.
    cos = (d[1:] * d[:-1]).sum(-1) / (norm[1:] * norm[:-1])
    return norm, cos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="SiT-L/2", choices=list(SiT_models))
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--weights-source", default="", help="where the weights came from, recorded in the manifest")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--steps", type=int, default=50, help="grid points; Euler makes steps-1 model calls")
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--device", default="mps")
    a = p.parse_args()

    torch.set_grad_enabled(False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(a.model, a.weights, a.device)
    # Linear path + velocity prediction, with None for the rest, are the sample.py defaults. The L/2 card does not state its training setup.
    # Euler makes exactly one model call per grid interval. atol and rtol are required by the wrapper but do nothing here.
    sample_fn = Sampler(create_transport("Linear", "velocity", None, None, None)).sample_ode(
        sampling_method="euler", num_steps=a.steps, atol=1e-6, rtol=1e-3, reverse=False)

    runs = []
    for seed in a.seeds:
        # Defensive only: Euler sampling in eval mode draws no random numbers. The noise comes from initial_noise().
        torch.manual_seed(seed)
        start = time.perf_counter()
        z, final, log = run_seed(model, sample_fn, seed, a.cfg, a.device)
        # One call per interval is what makes step k and time t[k] line up. An extra call would silently misalign every metric.
        assert len(log["t"]) == a.steps - 1, f"expected {a.steps - 1} model calls, got {len(log['t'])}"
        cond, uncond = torch.stack(log["cond"]).numpy(), torch.stack(log["uncond"]).numpy()
        delta = cond - uncond
        norm, cos = metrics(delta)
        # forward_with_cfg applies guidance to 3 of the 4 channels, so also measure the part of Delta the sampler really uses.
        norm_guided, cos_guided = metrics(delta[:, :, :GUIDED_CHANNELS])
        assert np.isfinite(final).all() and np.isfinite(delta).all(), f"non-finite values at seed {seed}"
        # Saving Delta, v_uncond and x together is enough to rebuild v_cond, replay the sampler, and convert to other parameterizations.
        # Axes: t (T,); x, delta, v_uncond (T, 8, 4, 32, 32); norm (T, 8); cos (T-1, 8); z, final (8, 4, 32, 32).
        np.savez(out / f"seed_{seed}.npz", seed=seed, labels=np.array(LABELS), t=np.array(log["t"]), z=z, final=final,
                 x=torch.stack(log["x"]).numpy(), delta=delta, v_uncond=uncond, norm=norm, cos=cos,
                 norm_guided=norm_guided, cos_guided=cos_guided)
        seconds = time.perf_counter() - start
        # These hashes let another run prove it used the same noise and reached bit-identical samples. seconds includes saving.
        runs.append(dict(seed=seed, seconds=round(seconds, 1), noise_sha256=array_sha256(z), final_sha256=array_sha256(final)))
        print(f"seed {seed}: {seconds:.1f}s mean|delta|={norm.mean():.3f} mean cos={cos.mean():.4f}", flush=True)

    # Written only after every seed finishes, so a crashed run leaves .npz files with no manifest.
    commit = subprocess.run(["git", "-C", "sit", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", "sit", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    packages = "torch torchvision torchdiffeq diffusers timm numpy safetensors".split()
    manifest = dict(
        command=" ".join(sys.argv), model=a.model, weights=a.weights, weights_sha256=sha256(a.weights), weights_source=a.weights_source,
        seeds=a.seeds, seed_use="torch.Generator().manual_seed(seed) on CPU; torch.randn(8, 4, 32, 32)",
        labels=LABELS, label_names=LABEL_NAMES, null_class=NULL_CLASS, conditioning="ImageNet class label (SiT has no text prompts)",
        cfg_scale=a.cfg, guided_channels=GUIDED_CHANNELS, parameterization="velocity (native model output)",
        delta="v_cond - v_uncond, both evaluated at the guided x_t, all 4 latent channels",
        path_type="Linear", prediction="velocity", solver="torchdiffeq euler via Sampler.sample_ode", grid_points=a.steps,
        model_calls=a.steps - 1, time_convention="t=0 noise, t=1 data", device=a.device, dtype="float32",
        sit_commit=commit, sit_dirty=bool(dirty), python=platform.python_version(), platform=platform.platform(),
        machine=platform.machine(), packages={k: md.version(k) for k in packages}, runs=runs)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
