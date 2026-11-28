"""Do the extra variation directions that condition damping adds in the late blocks cause its semantic diversity? (direction 3)

RESULTS.md 11.2: after the window, condition damping raises the effective rank of the late blocks' pooled features (blocks
13-23: 7.25 at w = 4, 8.49 at cond_1) and recovers semantic diversity; score damping restores spread in few directions and
little semantic diversity. That is a correlation. This tests it causally.

Fit (--fit, CPU; mechanistic run, seeds 16-31). Per late block b and class c, from the conditional pooled features after the
window, centered per step over seeds:
  S    top S_RANK directions of seed variation at w = 4
  E    "extra" directions: top RANK directions of cond_1's variation after projecting out S
  O    control: top RANK directions of cond_1's variation inside S (directions w = 4 also varies along)
  m    cond_1's mean pooled feature over seeds, per step, block, class and branch: the reference point for coordinates
Sample (--sample, GPU; seeds 32-47, never used to fit). Hooks on blocks 13-23 act only after the window (s_t >= 0.3), where
cond_1 no longer edits the network, on both branches. A sample's coordinates along E are (pooled - m) . E:
  cond_1+rec      cond_1, recording each sample's coordinates along E (must reproduce cond_1 bit for bit)
  cond_1+xablate  cond_1, every token loses (coordinates along E) . E: the extra variation is removed
  cond_1+oablate  the same with O (control: equal rank, directions w = 4 already varies along)
  w4+xinject      w = 4, every token shifted so its coordinates along E equal that seed's cond_1+rec coordinates
Then guidance_intervene_eval.py --embed and paired_vs_cond.py / points.py on the run (baselines symlinked from SiT-L_2_confirm).
usage: extra_dirs.py --fit --mech <mech run> --tables <out.npz>
       extra_dirs.py --sample --tables <npz> --itables <intervention tables> --weights <file> --out <run> --seeds 32 ... --conditions ...
"""
import argparse, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, "bench")
from common import LABELS, array_sha256

BLOCKS = list(range(13, 24))
RANK, S_RANK = 3, 8
EARLY_END = 0.3


def fit(mech, path, seeds=range(16, 32)):
    mech = Path(mech)
    mm = {c: [np.load(mech / c / f"seed_{s}" / "pooled.npy", mmap_mode="r") for s in seeds] for c in ("w4", "cond_1")}
    s_t = np.load(mech / "w4" / f"seed_{seeds[0]}" / "trajectory.npz")["s_t"]
    late = np.flatnonzero(s_t >= EARLY_END)
    E = np.zeros((len(BLOCKS), 8, RANK, 1024), np.float32); O = np.zeros_like(E)
    m = np.zeros(mm["cond_1"][0][:, BLOCKS].shape, np.float64)  # (49, 11, 2, 8, 1024); a running sum keeps one seed in memory
    for p in mm["cond_1"]:
        m += p[:, BLOCKS]
    m = (m / len(seeds)).astype(np.float32)
    stats = []
    for j, b in enumerate(BLOCKS):
        X = {c: np.stack([np.asarray(p[late, b, 0], dtype=np.float64) for p in mm[c]]) for c in mm}  # (16, 34, 8, 1024)
        for i in range(8):
            dev = {c: (X[c][:, :, i] - X[c][:, :, i].mean(0)).reshape(-1, 1024) for c in X}  # centered per step over seeds
            S = np.linalg.svd(dev["w4"], full_matrices=False)[2][:S_RANK]
            outside = dev["cond_1"] - dev["cond_1"] @ S.T @ S
            E[j, i] = np.linalg.svd(outside, full_matrices=False)[2][:RANK]
            O[j, i] = (np.linalg.svd(dev["cond_1"] @ S.T, full_matrices=False)[2][:RANK]) @ S  # top RANK directions inside span(S)
            var = lambda D, B: float(((D @ B.T) ** 2).sum() / (D ** 2).sum())  # share of D's variance along the rows of B
            stats.append((b, i, var(dev["cond_1"], E[j, i]), var(dev["cond_1"], O[j, i]), var(dev["w4"], E[j, i]), var(dev["w4"], O[j, i])))
    st = np.array(stats)
    print(f"share of seed variance, mean over late blocks and classes: cond_1 along E {st[:, 2].mean():.3f}, along O {st[:, 3].mean():.3f}; "
          f"w4 along E {st[:, 4].mean():.3f}, along O {st[:, 5].mean():.3f}")
    np.savez(path, E=E, O=O, m=m, blocks=np.array(BLOCKS), seeds=np.array(list(seeds)), rank=RANK, s_rank=S_RANK, stats=st)


def sample(a):
    import torch
    from common import load_model
    from guidance_intervene import load_tables, run_condition
    from transport import Sampler, create_transport
    torch.set_grad_enabled(False)
    dev = a.device
    t = np.load(a.tables)
    E, O, m = (torch.from_numpy(t[k]).to(dev) for k in ("E", "O", "m"))  # m: (49, 11 blocks, 2 branches, 8 classes, 1024)
    model = load_model("SiT-L/2", a.weights, dev)
    itables = load_tables(a.itables, dev)[0]
    sample_fn = Sampler(create_transport("Linear", "velocity", None, None, None)).sample_ode(
        sampling_method="euler", num_steps=50, atol=1e-6, rtol=1e-3, reverse=False)
    n = len(LABELS)
    cls = torch.arange(2 * n, device=dev) % n; br = torch.arange(2 * n, device=dev) // n  # row -> class, branch
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        for condition in a.conditions:
            base, mode = condition.split("+")
            target = out / condition / f"seed_{seed}.npz"
            if target.exists():
                continue
            target.parent.mkdir(exist_ok=True)
            state = dict(k=0)
            coords_log = np.zeros((49, len(BLOCKS), 2 * n, RANK), np.float32)
            if mode == "xinject":
                src = np.load(out / "cond_1+rec" / f"seed_{seed}.npz")["coords"]
                src = torch.from_numpy(src).to(dev)

            def t_hook(_m, inputs):
                state["k"] = int(round(float(inputs[0][0]) * 49))  # grid t = k / 49

            def block_hook(j):
                def hook(_m, _i, h):
                    k = state["k"]
                    if k / 49 < EARLY_END:
                        return None
                    basis = (O if mode == "oablate" else E)[j][cls]  # (2n, RANK, 1024)
                    ref = m[k, j][br, cls]  # (2n, 1024)
                    coords = torch.einsum("nd,nrd->nr", h.mean(1) - ref, basis)
                    if mode == "rec":
                        coords_log[k, j] = coords.cpu().numpy()
                        return None
                    change = -coords if mode in ("xablate", "oablate") else src[k, j] - coords
                    return h + torch.einsum("nr,nrd->nd", change, basis)[:, None, :]
                return hook

            handles = [model.t_embedder.register_forward_pre_hook(t_hook)]
            handles += [model.blocks[b].register_forward_hook(block_hook(j)) for j, b in enumerate(BLOCKS)]
            start = time.perf_counter()
            try:
                z, final = run_condition(model, sample_fn, base, seed, dev, itables)
            finally:
                for h in handles:
                    h.remove()
            extra = dict(coords=coords_log) if mode == "rec" else {}
            tmp = target.with_suffix(".tmp.npz")
            np.savez(tmp, seed=seed, z=z, final=final, final_sha256=array_sha256(final), **extra)
            tmp.rename(target)
            print(f"{condition} seed={seed}: {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fit", action="store_true"); p.add_argument("--sample", action="store_true")
    p.add_argument("--mech"); p.add_argument("--tables", required=True); p.add_argument("--itables")
    p.add_argument("--weights"); p.add_argument("--out"); p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--conditions", nargs="+", default=["cond_1+rec", "cond_1+xablate", "cond_1+oablate", "w4+xinject"])
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    if a.fit:
        fit(a.mech, a.tables)
    if a.sample:
        sample(a)
