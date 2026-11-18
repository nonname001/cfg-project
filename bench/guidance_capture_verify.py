"""Check a guidance_capture.py run. Prints one line per check and exits non-zero if any fails.

Cheap checks run on every finished batch; --deep also recomputes one batch's activations with an independent CPU forward pass.
usage: guidance_capture_verify.py --run <dir> [--first-run <dir of guidance_delta.py>] [--deep <w or condition name> seed]

What each check rules out:
  same / different noise    a condition accidentally starting from other noise, which would make paired comparisons meaningless
  bit-identical to first    the logging hooks changing the samples they were meant only to observe
  Euler replay              logged f_guided not being the velocity the solver actually integrated
  x_hat definition          a mistake in the stored predicted clean latent
  CFG identity              f_guided not equal to f_uncond + w * Delta (for runs without interventions)
  w=1 uses f_cond           the windowing logic leaving guidance on where it should be off
  pooled = token mean       the pooled and token logs disagreeing, e.g. from mixing up steps or layers
  --deep                    the hooks reading the wrong tensors: recomputes everything with a hand-written CPU forward pass
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "bench")
from guidance_delta import LABELS, NULL_CLASS, load_model

failures = []  # names of failed checks; a non-empty list makes the script exit with status 1


def check(name, ok, detail=""):
    """Print PASS/FAIL for one check and remember failures."""
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        failures.append(name)


def batches(run):
    """Yield (metadata, folder) for every finished batch in the run, under any condition."""
    for path in sorted(run.glob("*/seed_*/metadata.json")):
        yield json.loads(path.read_text()), path.parent


def replay_error(d, grid):
    """Largest error when rebuilding every Euler step from the logged guided velocity."""
    # All T+1 states: the state before each model call, then the final state.
    states = np.concatenate([d["x_t"], d["final"][None]], 0)
    dt = (grid[1:] - grid[:-1])[:, None, None, None, None]
    # Euler: x[k+1] = x[k] + dt * f_guided[k]. The error should be float32 rounding only.
    return float(np.abs(states[:-1] + dt * d["f_guided"] - states[1:]).max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--first-run")
    p.add_argument("--deep", nargs=2)
    a = p.parse_args()
    run = Path(a.run)
    man = json.loads((run / "manifest.json").read_text())
    T, g = man["model_calls"], man["guided_channels"]
    grid = np.linspace(0, 1, man["grid_points"], dtype=np.float32)  # the same float32 time grid the sampler used
    seen = list(batches(run))
    print(f"{len(seen)} batches", flush=True)

    # Group the noise hashes by seed: one seed must give one hash in every condition, and different seeds different hashes.
    noise = {}
    for m, _ in seen:
        noise.setdefault(m["seed"], set()).add(m["noise_sha256"])
    check("same noise across conditions", all(len(v) == 1 for v in noise.values()), f"{len(noise)} seeds")
    check("different noise across seeds", len({next(iter(v)) for v in noise.values()}) == len(noise))

    if a.first_run:
        # guidance_delta.py had only the minimal wrapper. If this run's samples at the same w and seed are bit-identical to that
        # run's, the extra block hooks here changed nothing about sampling.
        first = json.loads((Path(a.first_run) / "manifest.json").read_text())
        old = {r["seed"]: r["final_sha256"] for r in first["runs"]}
        pairs = [(m["seed"], m["final_sha256"] == old[m["seed"]]) for m, _ in seen if m["w"] == first["cfg_scale"] and m["seed"] in old]
        check("hooks did not change samples: bit-identical to first run", pairs and all(ok for _, ok in pairs), f"{len(pairs)} seeds")

    # Per-batch maximum errors, collected over the whole run, then checked once each below.
    replay, xhat, drift, pool, w1 = [], [], [], [], []
    for m, path in seen:
        d = dict(np.load(path / "trajectory.npz"))
        assert d["x_t"].shape == (T, 8, 4, 32, 32) and d["s_t"].shape == (T,) and d["cos"].shape == (T - 1, 8)
        replay.append(replay_error(d, grid))
        # Recompute x_hat with the same float32 expression; it must match exactly (max error 0).
        xhat.append(float(np.abs(d["x_hat"] - (d["x_t"] + (1 - d["s_t"]).astype(np.float32)[:, None, None, None, None] * d["f_guided"])).max()))
        # Rebuild what forward_with_cfg should have returned from the two logged branches.
        expected = d["f_cond"].copy()  # channel 3, and every channel when w=1
        w_t = d["w_t"] if "w_t" in d else np.full(T, m["w"])  # runs without a window have one scale throughout
        expected[:, :, :g] = d["f_uncond"][:, :, :g] + w_t.astype(np.float32)[:, None, None, None, None] * d["delta"][:, :, :g]
        if not m.get("intervention"):  # an intervention changes f_guided on purpose; the Euler replay still checks it
            drift.append(float(np.abs(expected - d["f_guided"]).max()))
        if m["w"] == 1 and not m.get("intervention"):
            w1.append(float(np.abs(d["f_guided"] - d["f_cond"]).max()))
        pooled = np.load(path / "pooled.npy", mmap_mode="r")
        tokens = np.load(path / "tokens.npy", mmap_mode="r")
        assert pooled.shape == (T, 24, 2, 8, 1024) and tokens.shape == (10, 3, 2, 8, 256, 1024)
        # Wherever both were logged, the mean of the tokens must equal the pooled vector.
        for i, step in enumerate(man["token_steps"]):
            for j, layer in enumerate(man["token_layers_zero_indexed"]):
                pool.append(float(np.abs(np.asarray(tokens[i, j]).mean(2) - pooled[step, layer]).max()))
    # Tolerances sit well above float32 rounding (~1e-7 relative) yet far below the size of any real bookkeeping error.
    check("Euler replay from logged f_guided", max(replay) < 1e-5, f"max {max(replay):.2e}")
    check("x_hat matches definition", max(xhat) == 0, f"max {max(xhat):.2e}")
    if drift:
        check("f_guided = CFG of f_cond, f_uncond (no intervention)", max(drift) < 1e-4, f"max {max(drift):.2e}")
    if w1:
        check("w=1 sampler used f_cond", max(w1) < 1e-4, f"max {max(w1):.2e}")
    check("pooled = token mean at shared steps and layers", max(pool) < 1e-3, f"max {max(pool):.2e}")

    if a.deep:
        w, seed = a.deep
        folder = f"w{w}" if w.replace(".", "").isdigit() else w  # a bare number means plain guidance, w<number>
        path = run / folder / f"seed_{seed}"
        d = np.load(path / "trajectory.npz")
        pooled, tokens = np.load(path / "pooled.npy", mmap_mode="r"), np.load(path / "tokens.npy", mmap_mode="r")
        torch.set_grad_enabled(False)
        model = load_model(man["checkpoint"]["model"], man["checkpoint"]["path"], "cpu")
        slot = 5  # the 6th logged token step (step 27 of 49), mid-trajectory
        step = man["token_steps"][slot]
        x = torch.from_numpy(d["x_t"][step])
        xb, y = torch.cat([x, x]), torch.tensor(LABELS + [NULL_CLASS] * 8)
        t = torch.full((16,), float(d["s_t"][step]))
        # The forward pass written out by hand: no hooks, no wrapper, and on a different device.
        # This mirrors SiT.forward: patch embedding + position, time + label conditioning, the blocks, then the output layer.
        # The step checked, token step 5, sits at s_t = 27/49 ≈ 0.55, after the s_t < 0.3 intervention window. So for intervention
        # runs too, no hook was active there and the plain model must reproduce the logged values.
        h = model.x_embedder(xb) + model.pos_embed
        c = model.t_embedder(t) + model.y_embedder(y, False)
        for i, block in enumerate(model.blocks):
            h = block(h, c)
            ref = h.mean(1).numpy().reshape(2, 8, -1)
            err = np.linalg.norm(ref - pooled[step, i]) / np.linalg.norm(ref)  # relative error of the logged pooled vector
            if i in man["token_layers_zero_indexed"]:
                j = man["token_layers_zero_indexed"].index(i)
                ref_tokens = h.numpy().reshape(2, 8, 256, -1)
                terr = np.linalg.norm(ref_tokens - tokens[slot, j]) / np.linalg.norm(ref_tokens)
                check(f"tokens block {i} step {step} vs CPU", terr < 1e-3, f"rel err {terr:.2e}")
                check(f"pooled block {i} step {step} vs CPU", err < 1e-3, f"rel err {err:.2e}")
        # [:, :4] keeps the 4 velocity channels and drops the 4 learned-sigma channels, as SiT.forward does.
        v = model.unpatchify(model.final_layer(h, c))[:, :4].numpy()
        for name, branch in (("f_cond", v[:8]), ("f_uncond", v[8:])):
            err = np.linalg.norm(branch - d[name][step]) / np.linalg.norm(branch)
            check(f"{name} step {step} vs CPU", err < 1e-3, f"rel err {err:.2e}")

    print("ALL PASS" if not failures else f"FAILED: {failures}", flush=True)
    sys.exit(bool(failures))


if __name__ == "__main__":
    main()
