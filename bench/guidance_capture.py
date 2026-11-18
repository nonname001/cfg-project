"""Sample paired CFG runs and log each branch's velocity and each block's pooled activations at every step.

Each (condition, seed) batch samples the same 8 classes from the same noise, using fixed-step Euler through the unmodified
Sampler.sample_ode + model.forward_with_cfg. Nothing in sit/ is edited: the branch velocities come from an instance-level wrapper
on model.forward, and the activations from forward hooks on model.blocks.

Layout under --out:
    manifest.json                       settings shared by every batch; written before sampling starts
    <condition>/seed_<n>/metadata.json  per-batch metadata, noise and final-latent hashes, timing
    <condition>/seed_<n>/trajectory.npz s_t (T,); z, final (8,4,32,32); f_cond, f_uncond, delta (T,8,4,32,32)
    <condition>/seed_<n>/pooled.npy     (T, 24 blocks, 2 branches, 8 classes, 1024), each block's output averaged over 256 tokens

T = 49 model calls. Branch axis: 0 = conditional, 1 = unconditional. Rerunning with the same --out skips finished batches.
Conditions:
    --ws 4 1          plain guidance at each w, stored as w4/ and w1/
    --window lo hi    guidance w only for lo <= s_t < hi and 1 elsewhere (windowed guidance)
    --conditions ...  each name is w<value> or an intervention spec from guidance_intervene.py (score_1, cond_0.5,
                      score_1+cond_0.5), run at w = 4 and stored under that name
usage: guidance_capture.py --weights <file> --out <dir> [--seeds 0 1 ...] [--ws 4 1] [--window lo hi]    (from the project root)

Names used below:
    f_cond     velocity the network predicts with the class label (conditional branch)
    f_uncond   velocity it predicts with the null label (unconditional branch)
    delta      f_cond - f_uncond, the guidance vector
    pooled     each block's output averaged over all image tokens, one 1024-d vector per (block, branch, class)
"""
import argparse, json, platform, shutil, subprocess, sys, time
import importlib.metadata as md
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "bench")
# common puts sit/ on sys.path when imported, which is why `transport` imports below without its own insert.
from common import GUIDED_CHANNELS, LABEL_NAMES, LABELS, NULL_CLASS, array_sha256, initial_noise, load_model, sha256
from transport import Sampler, create_transport

SAMPLER = "euler"  # torchdiffeq fixed-step; SiT's default ODE sampler is adaptive dopri5, which has no fixed "step k"


def run_batch(model, sample_fn, seed, w, device, model_calls, window=None, intervention=None):
    """Sample one batch at guidance scale w, optionally only inside window, and return the logged arrays and pooled activations."""
    n = len(LABELS)
    z = initial_noise(seed)  # depends only on the seed, so every condition starts from identical noise
    # CFG batch layout: rows [:n] carry the real class labels, rows [n:] the null class, and both start from the same z.
    # forward_with_cfg reads only the first half of x and duplicates it, so both branches always see the same x_t.
    x = torch.cat([z, z], 0).to(device)
    y = torch.tensor(LABELS + [NULL_CLASS] * n, device=device)
    # Read the sizes from the model rather than hard-coding them: 24 blocks of width 1024 for SiT-L/2.
    n_blocks, width = len(model.blocks), model.blocks[0].norm1.normalized_shape[0]

    pooled = np.empty((model_calls, n_blocks, 2, n, width), np.float32)  # filled in place, one step at a time
    log = dict(s_t=[], f_cond=[], f_uncond=[])
    step = 0  # index of the model call in progress; advanced by recording_forward after each call
    last = {}  # the most recent raw branch output, still on the device, for the score intervention
    pooled_step = [None] * n_blocks  # stays on the GPU until the step ends, avoiding a device sync per block

    def block_hook(index):
        # A factory, so each block's hook remembers its own index. (A bare lambda in a loop would see only the last index.)
        def hook(_module, _inputs, out):  # out: (2n, tokens, width). Returning None leaves the output untouched.
            pooled_step[index] = out.mean(dim=1)  # the tokens are the 16x16 patch grid, so this averages over H and W
        return hook

    # Keep a reference to the real bound method before shadowing it, so the wrapper below can still call it.
    forward = model.forward

    def recording_forward(xb, tb, yb):
        # forward_with_cfg calls self.forward(...) directly, so ordinary nn.Module forward hooks on the model never fire.
        # Assigning this wrapper to model.forward on the instance shadows the class method and captures both branches.
        nonlocal step
        out = forward(xb, tb, yb)
        last["out"] = out
        log["s_t"].append(float(tb[0]))  # every row of tb holds the same time, so one value describes the step
        log["f_cond"].append(out[:n].float().cpu())  # raw branch outputs, before guidance mixes them
        log["f_uncond"].append(out[n:].float().cpu())
        # By now every block hook of this call has run, so all 24 pooled vectors exist and can be copied off the GPU in one go.
        # The 2n batch rows reshape to (2 branches, n classes): conditional rows come first, matching the branch axis.
        pooled[step] = torch.stack(pooled_step).float().cpu().numpy().reshape(n_blocks, 2, n, width)
        step += 1
        return out

    def guided(xb, tb, **kwargs):
        # The function the ODE solver integrates in place of model.forward_with_cfg. It can change the guidance scale per step
        # (windowed guidance) and edit the guided velocity afterwards (score-space interventions).
        if window and not window[0] <= float(tb[0]) < window[1]:
            kwargs["cfg_scale"] = 1.0  # outside the window: forward_with_cfg then returns f_cond on every channel
        if intervention:
            intervention.set_time(float(tb[0]))  # hooks inside the forward pass read this
        out = model.forward_with_cfg(xb, tb, **kwargs)
        if intervention:
            # The raw branch outputs of this step, which recording_forward just logged, rebuild Delta for the score part.
            out = intervention.adjust(out, last["out"])
        return out

    # Install all instrumentation: a forward hook on every transformer block, the forward shadow, and any intervention hooks.
    handles = [block.register_forward_hook(block_hook(i)) for i, block in enumerate(model.blocks)]
    model.forward = recording_forward
    if intervention:
        intervention.install(model)
    try:
        # sample_fn returns the state at every grid point; [-1][:n] is the final state of the conditional rows, the real samples.
        final = sample_fn(x, guided, y=y, cfg_scale=w)[-1][:n].float().cpu().numpy()
    finally:
        # Always undo the instrumentation, even after an error, so the model object is clean for the next batch.
        del model.__dict__["forward"]
        if intervention:
            intervention.remove()
        for handle in handles:
            handle.remove()

    # Exactly one model call per grid interval is what lines up step k with time s_t[k]; anything else would misalign every array.
    assert step == model_calls, f"expected {model_calls} model calls, got {step}"
    # Turn the per-step lists into arrays: s_t becomes (T,) and the velocities stack to (T, 8, 4, 32, 32).
    arrays = {k: np.array(v) if k == "s_t" else torch.stack(v).numpy() for k, v in log.items()}
    arrays["delta"] = arrays["f_cond"] - arrays["f_uncond"]
    arrays.update(z=z.numpy(), final=final)
    assert all(np.isfinite(a).all() for a in (*arrays.values(), pooled)), f"non-finite values: w={w} seed={seed}"
    return arrays, pooled


def write_batch(out, w, seed, arrays, pooled, seconds, checkpoint, window=None, name=None, intervention=None):
    """Write into a temporary folder, then rename: a batch folder is either complete or absent, even if the run is stopped."""
    name = name or f"w{w:g}"  # :g drops a trailing ".0", so w=4.0 is stored as "w4"
    final_dir = out / name / f"seed_{seed}"
    tmp = out / f".tmp_{name}_seed_{seed}"
    shutil.rmtree(tmp, ignore_errors=True)  # leftovers from a run that was stopped mid-write
    tmp.mkdir(parents=True)
    np.savez(tmp / "trajectory.npz", **arrays)
    np.save(tmp / "pooled.npy", pooled)  # plain .npy (not .npz) so readers can memory-map it with mmap_mode="r"
    # The noise and final hashes let a later run prove it saw the same noise and produced bit-identical samples.
    meta = dict(seed=seed, w=w, classes=LABELS, class_names=LABEL_NAMES, sampler=SAMPLER, model_calls=len(arrays["s_t"]),
                checkpoint=checkpoint, guided_channels=GUIDED_CHANNELS, noise_sha256=array_sha256(arrays["z"]),
                final_sha256=array_sha256(arrays["final"]), seconds=round(seconds, 1), window=window, condition=name, intervention=intervention)
    (tmp / "metadata.json").write_text(json.dumps(meta, indent=2))
    final_dir.parent.mkdir(exist_ok=True)
    # A rename within one filesystem is atomic: other code sees either no folder or the complete one, never a partial write.
    tmp.rename(final_dir)


def main():
    """Parse options, write or check the manifest, then sample every (seed, condition) batch that is not already on disk."""
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

    torch.set_grad_enabled(False)  # inference only: autograd would just waste memory
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model_calls = a.steps - 1
    checkpoint = dict(path=a.weights, sha256=sha256(a.weights), source=a.weights_source, model=a.model)
    commit = subprocess.run(["git", "-C", "sit", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", "sit", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    # Everything that must be identical for two batches to be comparable: the reproducibility record of the run. It leaves out
    # per-batch values (seed, w, timing), so a resumed run can check that it uses the same settings as the batches already saved.
    manifest = dict(
        checkpoint=checkpoint, classes=LABELS, class_names=LABEL_NAMES, null_class=NULL_CLASS,
        seed_use="torch.Generator().manual_seed(seed) on CPU; torch.randn(8, 4, 32, 32); identical across w",
        w_definition="f_uncond + w * (f_cond - f_uncond), on the first guided_channels channels", guided_channels=GUIDED_CHANNELS,
        unguided_channel_behavior="channel 3 always takes f_cond; at w=1 all channels equal f_cond",
        parameterization="velocity (native model output)", path_type="Linear", prediction="velocity", sampler=SAMPLER,
        deterministic=True, grid_points=a.steps, model_calls=model_calls, time_convention="s_t=0 noise, s_t=1 data",
        activation_site="output of model.blocks[i], before final_layer",
        branch_axis="0 = conditional, 1 = unconditional", device=a.device, dtype="float32", sit_commit=commit, sit_dirty=bool(dirty),
        python=platform.python_version(), platform=platform.platform(), machine=platform.machine(),
        packages={k: md.version(k) for k in "torch torchvision torchdiffeq diffusers timm numpy safetensors".split()})
    if a.window:  # only windowed runs carry this key, so manifests of unwindowed runs stay comparable
        manifest["window"] = a.window
    tables = None
    if a.conditions:
        # Imported only when needed: plain-w runs do not depend on the intervention code or its tables.
        from guidance_intervene import EARLY_END, RANK, W as INTERVENTION_W, Intervention, load_tables
        tables, calibration_seeds = load_tables(a.tables, a.device)
        # The tables were fit on some seeds; evaluating on those same seeds would let the method see its own training data.
        assert not set(a.seeds) & set(calibration_seeds), "evaluation seeds overlap the calibration seeds of the tables"
        manifest.update(tables=a.tables, tables_sha256=sha256(a.tables), intervention_w=INTERVENTION_W, early_end=EARLY_END, rank=RANK,
                        calibration_seeds=calibration_seeds)
    path = out / "manifest.json"
    if path.exists():  # a resumed run must match the settings of the batches already on disk
        assert json.loads(path.read_text()) == manifest, "settings differ from the existing manifest in --out"
    else:
        path.write_text(json.dumps(manifest, indent=2))

    model = load_model(a.model, a.weights, a.device)
    # Linear path + velocity prediction are SiT's sample.py defaults. Euler makes exactly one model call per grid interval;
    # atol and rtol are required by the wrapper but unused by Euler.
    sample_fn = Sampler(create_transport("Linear", "velocity", None, None, None)).sample_ode(
        sampling_method=SAMPLER, num_steps=a.steps, atol=1e-6, rtol=1e-3, reverse=False)

    # Build the plan: a list of (folder name, guidance scale, intervention spec or None).
    # "w4" -> plain guidance at 4.0; anything else (e.g. "score_1") -> that intervention, run at the intervention scale W = 4.
    if a.conditions:
        plan = [(c, float(c[1:]), None) if c.startswith("w") else (c, INTERVENTION_W, c) for c in a.conditions]
    else:
        plan = [(f"w{w:g}", w, None) for w in a.ws]
    for seed in a.seeds:  # conditions are the inner loop, so stopping early leaves complete sets for the finished seeds
        for name, w, spec in plan:
            # metadata.json exists only inside a fully renamed batch folder, so its presence means "this batch is done".
            if (out / name / f"seed_{seed}" / "metadata.json").exists():
                print(f"skip {name} seed={seed}", flush=True)
                continue
            start = time.perf_counter()
            # A fresh Intervention per batch, so no state (time flag, hook handles) leaks from one batch into the next.
            intervention = Intervention(spec, tables, len(LABELS)) if spec else None
            arrays, pooled = run_batch(model, sample_fn, seed, w, a.device, model_calls, a.window, intervention)
            write_batch(out, w, seed, arrays, pooled, time.perf_counter() - start, checkpoint, a.window, name, spec)
            print(f"{name} seed={seed}: {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
