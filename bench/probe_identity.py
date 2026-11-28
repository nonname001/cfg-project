"""Where and when does a sample's final semantic identity become linearly readable inside SiT? (open direction 2b)

Sampling is deterministic, so the initial noise fixes the final image; the question is when the network's own state makes it
*linearly* readable. Target: the final image's DINOv2 embedding (unit length) minus its class mean over seeds, i.e. where the
sample sits within its class. Features at each step k, from the same run:
  block b      pooled output of block b (conditional branch), class-centered
  x_t          the latent state itself (4 x 32 x 32)
  dino(x_hat)  DINOv2 embedding of the decoded predicted clean image at step k (what the model is currently aiming for)
Probe: dual-form ridge regression (128 samples = 16 seeds x 8 classes, all classes pooled), alpha chosen per training fold by
a 3-fold cross-validation over the training seeds, evaluated by 4-fold cross-validation over seeds (4 held-out seeds per fold, every class). Class
means are taken from the training seeds only. Target dimensions: the top KPC = 8 directions of the final embedding's within-class variation, fit on the training seeds
(a 768-d target cannot be learned from 96 samples; most of its dimensions carry little variance). Score: R^2 = 1 - SSE / SST on
the held-out seeds, SST around the training class
mean; 0 = no better than the class mean, 1 = identity fully read.
Run on w = 1 and w = 4 separately (each predicts its own final image). Also "cosine to final": the mean cosine between
dino(x_hat_k) and the final DINOv2 embedding, after removing the class mean (an output-side, probe-free check).
usage: probe_identity.py <16-seed run>   -> <run>/analysis/probe_identity.csv
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import sys
sys.path.insert(0, "bench")
import numpy as np
from pathlib import Path
from guidance_analysis import unit, write_csv

SEEDS = range(16); WS = ("w1", "w4")
ALPHAS = np.logspace(-3, 4, 15)
KPC = 8  # target: the final embedding's top 8 within-class directions (where samples of a class mainly differ)
FOLDS = [np.arange(f, 16, 4) for f in range(4)]  # held-out seeds per fold: {0,4,8,12}, {1,5,9,13}, ...


def load_embeddings(run):
    """DINOv2 embeddings of x_hat at every step and of the final image: (16 seeds, 50, 8 classes, 768) per w."""
    return {w: np.stack([np.load(run / "analysis" / "embeddings" / f"{w}_seed_{s}.npz")["dino"] for s in SEEDS]).astype(np.float64) for w in WS}


def ridge_all(Xtr, Ytr, Xte):
    """Dual ridge predictions on Xte for every alpha in ALPHAS (relative to the kernel's mean eigenvalue): (alphas, rows, out)."""
    lam, U = np.linalg.eigh(Xtr @ Xtr.T)
    lam = np.clip(lam, 0, None)
    UtY, KteU = U.T @ Ytr, (Xte @ Xtr.T) @ U
    return np.stack([KteU @ (UtY / (lam + a)[:, None]) for a in ALPHAS * max(lam.mean(), 1e-12)])


def fold(X, Y, train, test, inner=True):
    """Center on the training seeds' class means, scale, and predict the test seeds. With inner=True, alpha is chosen by a
    3-fold cross-validation over the training seeds (GCV and leave-one-out are unreliable here, with features >> samples)."""
    mx, my = X[train].mean(0), Y[train].mean(0)
    flat = lambda A, m, s: (A[s] - m).reshape(-1, A.shape[-1])
    Xtr, Xte, Ytr, Yte = flat(X, mx, train), flat(X, mx, test), flat(Y, my, train), flat(Y, my, test)
    scale = np.sqrt((Xtr ** 2).sum(1).mean()) + 1e-12
    if not inner:
        return Xtr / scale, Ytr, Xte / scale, Yte
    err = np.zeros(len(ALPHAS))
    for j in range(3):
        itest, itrain = train[j::3], np.setdiff1d(train, train[j::3])
        a, b, c, d = fold(X, Y, itrain, itest, inner=False)
        err += ((ridge_all(a, b, c) - d[None]) ** 2).sum((1, 2))
    pred = ridge_all(Xtr / scale, Ytr, Xte / scale)[err.argmin()]
    return pred, Yte


def top_pcs(Y, train):
    """The KPC main directions of within-class variation of Y, from the training seeds only (all classes pooled)."""
    dev = (Y[train] - Y[train].mean(0)).reshape(-1, Y.shape[-1])
    return np.linalg.svd(dev, full_matrices=False)[2][:KPC]


def r2(X, Y, direct=False, folds=None):
    """X: (16 seeds, 8 classes, D) features, Y: (16, 8, 768) targets. 4-fold-over-seeds R^2 of the target's top KPC
    within-class directions (fit on the training seeds), class-centered on training seeds. direct=True: X lives in Y's space
    (dino(x_hat)) and the prediction is X itself, projected the same way and scaled by one least-squares factor."""
    sse = sst = 0.0
    folds = FOLDS if folds is None else folds  # index arrays into the seed axis of X and Y
    for test in folds:
        train = np.setdiff1d(np.arange(len(X)), test)
        V = top_pcs(Y, train)
        Yp = Y @ V.T
        if direct:
            Xp = X @ V.T
            c = lambda A: A - A[train].mean(0)
            xtr, ytr = c(Xp)[train], c(Yp)[train]
            a = (xtr * ytr).sum() / max((xtr ** 2).sum(), 1e-12)
            pred, Yte = a * c(Xp)[test].reshape(-1, KPC), c(Yp)[test].reshape(-1, KPC)
        else:
            pred, Yte = fold(X, Yp, train, test)
        sse += ((Yte - pred) ** 2).sum(); sst += (Yte ** 2).sum()
    return 1 - sse / sst


if __name__ == "__main__":
  run = Path(sys.argv[1])
  emb = load_embeddings(run)
  TARGET = {w: unit(emb[w][:, -1]) for w in WS}  # (16 seeds, 8 classes, 768)
  rows = []
  s_t = np.load(run / "w4" / "seed_0" / "trajectory.npz")["s_t"]
  pooled = {w: [np.load(run / w / f"seed_{s}" / "pooled.npy", mmap_mode="r") for s in SEEDS] for w in WS}
  traj = {w: [np.load(run / w / f"seed_{s}" / "trajectory.npz") for s in SEEDS] for w in WS}
  xt = {w: np.stack([t["x_t"] for t in traj[w]]).astype(np.float64) for w in WS}  # (16, 49, 8, 4, 32, 32): ~200 MB each
  for k in range(len(s_t)):
      for w in WS:
          Y = TARGET[w]
          row = dict(step=k, s_t=float(s_t[k]), w=w)
          P = np.stack([np.asarray(p[k, :, 0], dtype=np.float64) for p in pooled[w]])  # (16, 24 blocks, 8, 1024)
          for b in range(P.shape[1]):
              row[f"block_{b}"] = r2(P[:, b], Y)
          row["x_t"] = r2(xt[w][:, k].reshape(16, 8, -1), Y)
          row["dino_xhat"] = r2(emb[w][:, k], Y)
          row["dino_xhat_direct"] = r2(unit(emb[w][:, k]), Y, direct=True)
          a, f = unit(emb[w][:, k]), TARGET[w]
          a, f = a - a.mean(0), f - f.mean(0)
          row["cos_to_final"] = float((unit(a) * unit(f)).sum(-1).mean())
          rows.append(row)
      best = max(range(24), key=lambda b: rows[-1][f"block_{b}"])
      print(f"step {k:2d} s_t {s_t[k]:.2f}  " + "  ".join(
          f"{r['w']}: best block {max(range(24), key=lambda b: r[f'block_{b}'])} R2 {max(r[f'block_{b}'] for b in range(24)):.2f}, x_t {r['x_t']:.2f}, dino(x_hat) {r['dino_xhat']:.2f}/{r['dino_xhat_direct']:.2f}, cos {r['cos_to_final']:.2f}"
          for r in rows[-2:]), flush=True)
  write_csv(run / "analysis" / "probe_identity.csv", rows)
