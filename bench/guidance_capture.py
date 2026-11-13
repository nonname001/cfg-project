"""Log velocities, trajectories and internal activations for paired CFG runs at several guidance scales w.

Each (w, seed) batch samples the same 8 classes from the same noise for every w, using fixed-step Euler through the unmodified
Sampler.sample_ode + model.forward_with_cfg. Nothing in sit/ is edited: activations come from forward hooks on model.blocks,
and the branch velocities from the same instance-level wrapper on model.forward as guidance_delta.py.

Layout under --out:
    manifest.json                 settings shared by every run; written before sampling starts
    index.csv                     one row per (w, seed, class); rebuilt at the end of each invocation
    w<w>/seed_<n>/metadata.json   per-run metadata, hashes and timing
    w<w>/seed_<n>/trajectory.npz  s_t (T,); z, final (8,4,32,32); x_t, x_hat, f_cond, f_uncond, f_guided, delta (T,8,4,32,32);
                                  norm, norm_guided (T,8); cos, cos_guided (T-1,8)
    w<w>/seed_<n>/pooled.npy      (T, 24 blocks, 2 branches, 8 classes, 1024), mean over the 256 tokens
    w<w>/seed_<n>/tokens.npy      (10 timesteps, 3 layers, 2 branches, 8 classes, 256, 1024)

Branch axis: 0 = conditional, 1 = unconditional. Rerunning with the same --out skips finished batches.
With --window lo hi, guidance is w only for lo <= s_t < hi and 1 elsewhere; each step's scale is logged as w_t.
With --conditions, each name is either w<value> (plain guidance) or an intervention spec from guidance_intervene.py such as
score_1 or score_1+cond_0.5, run at w=4 and stored under that name instead of w<w>. Interventions also log `removed`.
usage: guidance_capture.py --weights <file> --out <dir> [--seeds 0 1 ...] [--ws 4 1] [--window lo hi]    (from the project root)
"""
import argparse, csv, json, platform, shutil, subprocess, sys, time
import importlib.metadata as md
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "bench")
from guidance_delta import (GUIDED_CHANNELS, LABEL_NAMES, LABELS, NULL_CLASS, array_sha256, initial_noise, load_model,
                            metrics, sha256)
from transport import Sampler, create_transport

TOKEN_LAYERS = [4, 12, 22]  # zero-indexed: model.blocks[i], so 22 is the 23rd of 24 blocks
N_TOKEN_STEPS = 10
SAMPLER = "euler"  # torchdiffeq fixed-step; SiT's default ODE sampler is adaptive dopri5


def token_steps(model_calls):
    """10 steps spread evenly over the model calls, rounded to the grid. 49 calls cannot divide exactly."""
    return np.linspace(0, model_calls - 1, N_TOKEN_STEPS).round().astype(int).tolist()


def run_batch(model, sample_fn, seed, w, device, model_calls, window=None, intervention=None):
    """Sample one batch at guidance scale w, optionally only inside window, and return every logged array."""
    n = len(LABELS)
    z = initial_noise(seed)  # depends only on the seed, so every w starts from identical noise
    x = torch.cat([z, z], 0).to(device)
    y = torch.tensor(LABELS + [NULL_CLASS] * n, device=device)
    steps_for_tokens = token_steps(model_calls)
    n_blocks, width = len(model.blocks), model.blocks[0].norm1.normalized_shape[0]
    n_tokens = model.x_embedder.num_patches

    pooled = np.empty((model_calls, n_blocks, 2, n, width), np.float32)
    tokens = np.empty((N_TOKEN_STEPS, len(TOKEN_LAYERS), 2, n, n_tokens, width), np.float32)
    log = dict(s_t=[], w_t=[], x_t=[], f_cond=[], f_uncond=[], f_guided=[])
    step = 0
    last = {}  # the most recent raw branch output, still on the device, for interventions that need it
    removed = []
    pooled_step = [None] * n_blocks  # stays on the GPU until the step ends, avoiding a device sync per block

    def block_hook(index):
        def hook(_module, _inputs, out):  # out: (2n, tokens, width). Returning None leaves the output untouched.
            pooled_step[index] = out.mean(dim=1)  # the tokens are the 16x16 patch grid, so this averages over H and W
            if index in TOKEN_LAYERS and step in steps_for_tokens:
                tokens[steps_for_tokens.index(step), TOKEN_LAYERS.index(index)] = \
                    out.float().cpu().numpy().reshape(2, n, n_tokens, width)
        return hook

    forward = model.forward

    def recording_forward(xb, tb, yb):
        # forward_with_cfg calls self.forward directly and passes cat([half, half]), so both branches share x_t.
        nonlocal step
        out = forward(xb, tb, yb)
        last["out"] = out
        log["s_t"].append(float(tb[0]))
        log["x_t"].append(xb[:n].float().cpu())
        log["f_cond"].append(out[:n].float().cpu())
        log["f_uncond"].append(out[n:].float().cpu())
        pooled[step] = torch.stack(pooled_step).float().cpu().numpy().reshape(n_blocks, 2, n, width)
        step += 1
        return out

    def guided(xb, tb, **kwargs):
        # Record what forward_with_cfg returned instead of re-deriving it, so f_guided is exactly what the solver integrated.
        if window and not window[0] <= float(tb[0]) < window[1]:
            kwargs["cfg_scale"] = 1.0  # outside the window: forward_with_cfg then returns f_cond on every channel
        log["w_t"].append(kwargs["cfg_scale"])
        if intervention:
            intervention.set_time(float(tb[0]))  # hooks inside the forward pass read this
        out = model.forward_with_cfg(xb, tb, **kwargs)
        if intervention:
            out, fraction = intervention.adjust(out, last["out"])
            removed.append(fraction)
        log["f_guided"].append(out[:n].float().cpu())
        return out

    handles = [block.register_forward_hook(block_hook(i)) for i, block in enumerate(model.blocks)]
    model.forward = recording_forward
    if intervention:
        intervention.install(model)
    try:
        final = sample_fn(x, guided, y=y, cfg_scale=w)[-1][:n].float().cpu().numpy()
    finally:
        del model.__dict__["forward"]
        if intervention:
            intervention.remove()
        for handle in handles:
            handle.remove()

    assert step == model_calls, f"expected {model_calls} model calls, got {step}"
    arrays = {k: np.array(v) if k in ("s_t", "w_t") else torch.stack(v).numpy() for k, v in log.items()}
    # Data prediction on the linear path x_t = s*x_data + (1-s)*noise, where v = x_data - noise: x_data = x_t + (1-s)*v.
    arrays["x_hat"] = arrays["x_t"] + (1 - arrays["s_t"]).astype(np.float32)[:, None, None, None, None] * arrays["f_guided"]
    arrays["delta"] = arrays["f_cond"] - arrays["f_uncond"]
    arrays["norm"], arrays["cos"] = metrics(arrays["delta"])
    arrays["norm_guided"], arrays["cos_guided"] = metrics(arrays["delta"][:, :, :GUIDED_CHANNELS])
    arrays.update(z=z.numpy(), final=final)
    if intervention:
        arrays["removed"] = np.array(removed)
    assert all(np.isfinite(a).all() for a in (*arrays.values(), pooled, tokens)), f"non-finite values: w={w} seed={seed}"
    return arrays, pooled, tokens


def write_batch(out, w, seed, arrays, pooled, tokens, seconds, checkpoint, window=None, name=None, intervention=None):
    """Write into a temporary folder, then rename: a batch folder is either complete or absent, even if the run is stopped."""
    name = name or f"w{w:g}"
    final_dir = out / name / f"seed_{seed}"
    tmp = out / f".tmp_{name}_seed_{seed}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    np.savez(tmp / "trajectory.npz", **arrays)
    np.save(tmp / "pooled.npy", pooled)
    np.save(tmp / "tokens.npy", tokens)
    meta = dict(seed=seed, w=w, classes=LABELS, class_names=LABEL_NAMES, sampler=SAMPLER, model_calls=len(arrays["s_t"]),
                checkpoint=checkpoint, guided_channels=GUIDED_CHANNELS, noise_sha256=array_sha256(arrays["z"]),
                final_sha256=array_sha256(arrays["final"]), seconds=round(seconds, 1), window=window, condition=name, intervention=intervention)
    (tmp / "metadata.json").write_text(json.dumps(meta, indent=2))
    final_dir.parent.mkdir(exist_ok=True)
    tmp.rename(final_dir)


def write_index(out, manifest):
    """One row per (w, seed, class), naming the files and the class's position on their batch axis."""
    with open(out / "index.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow("condition w seed class class_name batch_index sampler grid_points model_calls checkpoint_sha256 "
                        "guided_channels noise_sha256 run_dir".split())
        for path in sorted(out.glob("*/seed_*/metadata.json"), key=lambda p: (p.parts[-3], int(p.parts[-2][5:]))):
            m = json.loads(path.read_text())
            for i, (label, name) in enumerate(zip(m["classes"], m["class_names"])):
                writer.writerow([m.get("condition") or f"w{m['w']:g}", m["w"], m["seed"], label, name, i, m["sampler"], manifest["grid_points"], m["model_calls"],
                                 manifest["checkpoint"]["sha256"], m["guided_channels"], m["noise_sha256"],
                                 str(path.parent.relative_to(out))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="SiT-L/2")
    p.add_argument("--weights", required=True)
    p.add_argument("--weights-source", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(16)))
    p.add_argument("--ws", type=float, nargs="+", default=[4.0, 1.0])
    p.add_argument("--steps", type=int, default=50, help="grid points; Euler makes steps-1 model calls")
    p.add_argument("--device", default="mps")
    p.add_argument("--window", type=float, nargs=2, help="apply w only for lo <= s_t < hi; use 1 elsewhere")
    p.add_argument("--conditions", nargs="+", help="w<value> or an intervention spec; replaces --ws")
    p.add_argument("--tables", help="intervention tables from guidance_intervene.py --build-tables")
    a = p.parse_args()

    torch.set_grad_enabled(False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model_calls = a.steps - 1
    checkpoint = dict(path=a.weights, sha256=sha256(a.weights), source=a.weights_source, model=a.model)
    commit = subprocess.run(["git", "-C", "sit", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", "sit", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    manifest = dict(
        checkpoint=checkpoint, classes=LABELS, class_names=LABEL_NAMES, null_class=NULL_CLASS,
        seed_use="torch.Generator().manual_seed(seed) on CPU; torch.randn(8, 4, 32, 32); identical across w",
        w_definition="f_uncond + w * (f_cond - f_uncond), on the first guided_channels channels", guided_channels=GUIDED_CHANNELS,
        unguided_channel_behavior="channel 3 always takes f_cond; at w=1 all channels equal f_cond",
        parameterization="velocity (native model output)", path_type="Linear", prediction="velocity", sampler=SAMPLER,
        deterministic=True, grid_points=a.steps, model_calls=model_calls, time_convention="s_t=0 noise, s_t=1 data",
        x_hat="x_t + (1 - s_t) * f_guided, the predicted clean latent", token_layers_zero_indexed=TOKEN_LAYERS,
        token_steps=token_steps(model_calls), activation_site="output of model.blocks[i], before final_layer",
        branch_axis="0 = conditional, 1 = unconditional", device=a.device, dtype="float32", sit_commit=commit, sit_dirty=bool(dirty),
        python=platform.python_version(), platform=platform.platform(), machine=platform.machine(),
        packages={k: md.version(k) for k in "torch torchvision torchdiffeq diffusers timm numpy safetensors".split()})
    if a.window:  # only windowed runs carry this key, so manifests of unwindowed runs stay comparable
        manifest["window"] = a.window
    tables = None
    if a.conditions:
        from guidance_intervene import EARLY_END, RANK, W as INTERVENTION_W, Intervention, load_tables
        tables, calibration_seeds = load_tables(a.tables, a.device)
        assert not set(a.seeds) & set(calibration_seeds), "evaluation seeds overlap the calibration seeds of the tables"
        manifest.update(tables=a.tables, tables_sha256=sha256(a.tables), intervention_w=INTERVENTION_W, early_end=EARLY_END, rank=RANK,
                        calibration_seeds=calibration_seeds)
    path = out / "manifest.json"
    if path.exists():  # a resumed run must match the settings of the batches already on disk
        assert json.loads(path.read_text()) == manifest, "settings differ from the existing manifest in --out"
    else:
        path.write_text(json.dumps(manifest, indent=2))

    model = load_model(a.model, a.weights, a.device)
    sample_fn = Sampler(create_transport("Linear", "velocity", None, None, None)).sample_ode(
        sampling_method=SAMPLER, num_steps=a.steps, atol=1e-6, rtol=1e-3, reverse=False)

    if a.conditions:
        plan = [(c, float(c[1:]), None) if c.startswith("w") else (c, INTERVENTION_W, c) for c in a.conditions]
    else:
        plan = [(f"w{w:g}", w, None) for w in a.ws]
    for seed in a.seeds:  # conditions are the inner loop, so stopping early leaves complete sets for the finished seeds
        for name, w, spec in plan:
            if (out / name / f"seed_{seed}" / "metadata.json").exists():
                print(f"skip {name} seed={seed}", flush=True)
                continue
            start = time.perf_counter()
            intervention = Intervention(spec, tables, len(LABELS)) if spec else None
            arrays, pooled, tokens = run_batch(model, sample_fn, seed, w, a.device, model_calls, a.window, intervention)
            write_batch(out, w, seed, arrays, pooled, tokens, time.perf_counter() - start, checkpoint, a.window, name, spec)
            print(f"{name} seed={seed}: {time.perf_counter() - start:.1f}s mean|delta|={arrays['norm'].mean():.3f}", flush=True)
    write_index(out, manifest)


if __name__ == "__main__":
    main()
