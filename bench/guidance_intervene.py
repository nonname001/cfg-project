"""Damp the shared class component of guidance during early steps, in four spaces, at one compute budget.

Every condition runs the same Euler grid and 2 model passes per step, so all use 98 branch evaluations. w = 4 throughout except
where a condition changes guidance during s_t < EARLY_END. lambda is the damping strength; lambda = 1 aims to remove all excess
guidance along the targeted component, that is, guidance beyond w = 1. With excess guidance (w-1)*Delta, the methods are:

  scalar_L   w_early = 1 + (w-1)(1-L). Damps every direction equally: the control for whether targeting matters.
  score_L    f = f_cond + (w-1)(Delta - L P_c Delta), P_c = rank-3 projector onto class c's shared guidance subspace.
             The linear path makes score and velocity differ by a scalar per step, so this projection is the same in both.
  freq_L     as score_L, with P replaced by a low-pass filter at LOW_CUTOFF cycles. No calibration needed.
  cond_L     class embedding y' = null + a(y - null), a = 1 - L(w-1)/w, so that w*a = 1 at L=1 to first order.
  feat_L     in blocks 13-23, conditional rows lose (w-1)/w * L of the pooled cond-uncond difference inside a per-class rank-3
             subspace, again targeting w*(1 - (w-1)/w) = 1 to first order.
  wrongclass_L  score_L using the next class's subspace: tests whether class-specific directions matter.
  bcond_L0-L1-L2[-L3]  cond with its own strength per block group (BLOCK_GROUPS: blocks 0-7, 8-15, 16-23). The final layer
             takes L3 if given, else L2. L = 4/3 makes a = 0: those blocks see only the null class.
             Each group's blocks read c' = c_null + a_g (c_cond - c_null), a_g = 1 - L_g(w-1)/w. The timestep embedding is the
             same in both rows, so this equals cond's embedding interpolation, restricted to those blocks. bcond_L-L-L = cond_L.
  tfeat_L    in blocks 8-15, each conditional token loses (w-1)/w * L of its cond - uncond difference d inside a per-(block,
             class) rank-16 subspace: the top directions of d averaged over calibration seeds, at every token and early step.
             Unlike feat, it acts per token (most of d is not spatially uniform) and on the mid blocks, where d is shared
             across seeds. L = 4/3 removes the component entirely. Needs --token-tables (--build-token-tables).
  tfeatwrong_L  tfeat using the next class's bases: tests whether its class-specific directions matter.
  late_w     guidance scale w instead of W for s_t >= EARLY_END (after the window); combinable with the parts above.
  Modifiers of the token part: tfeatlate_L (also apply it after the window, at strength L), tend_t (token window s_t < t
  instead of EARLY_END), trank_k (top k of the TOKEN_RANK directions), tfrom_b (only blocks b and later of TOKEN_BLOCKS).
  cond_L can also join a combination (e.g. cond_1+late_6).
  smooth_d   replace the step at EARLY_END with a cosine ramp over EARLY_END +- d, for every damping strength and for late_w.
  cads_s     CADS: noisy class embedding, pure noise (rescaled) before s_t = 0.1, annealed to clean by 0.4; s = noise scale.
  cnoise_s   class embedding + s * std * fresh noise during the window; the class signal keeps its full strength.
  tshuffle_L as tfeat, but the subspace part is replaced by a seed-specific rotation of itself within the subspace.
  wall_w     guidance scale w at every step.
  slg_s      skip-layer guidance (blocks slgfrom_b..slgto_b, default 10-12, bypassed): f += s (f_cond - f_cond_skip) at every
             step. One extra forward pass on the conditional rows; cannot be combined with hook-based parts.
  mcond_L    damp the class dependence of adaLN modulation chunks (mchunks_<bitmask over MOD_CHUNKS>, default all) in
             blocks mfrom_<b>..mto_<b> (default all) during the window.
  A '+' joins at most one of score/wrongclass/freq, one of tfeat/tfeatwrong and one late_w, e.g. tfeat_1.3333+score_1+late_5.

score, freq and scalar act on the 3 guided channels only; cond and feat change f_cond, so they also reach channel 3.

Subspaces are fit on the calibration seeds of a guidance_capture.py run (--build-tables); evaluate on other seeds.
usage: guidance_intervene.py --build-tables --calibration <run> --tables <file>
       guidance_intervene.py --build-token-tables --calibration <token run> --token-tables <file>
       guidance_intervene.py --tables <file> [--token-tables <file>] --weights <file> --out <dir> --seeds 16 17 ... [--conditions w1 w4 ...]

Why "excess guidance": plain CFG gives f_guided = f_uncond + w*Delta = f_cond + (w-1)*Delta. The f_cond part is what w = 1 would
use, so (w-1)*Delta is everything guidance adds on top. Each method removes a fraction lambda of that excess along one component,
so lambda = 0 is plain w = 4 and lambda = 1 makes the targeted component behave as at w = 1.

Where each method acts, from the outside of the network inward:
  scalar, score, freq, wrongclass   after the network, on the velocity it returned (score space)
  cond                              before the blocks, on the class embedding that conditions every block (condition space)
  feat                              inside the network, on the outputs of blocks 13-23 (feature space)
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "bench")
from common import GUIDED_CHANNELS, LABELS, NULL_CLASS, array_sha256, initial_noise, load_model, sha256
from transport import Sampler, create_transport

W = 4.0  # guidance scale every intervention starts from
EARLY_END = 0.3  # interventions act only while s_t < 0.3, the early phase where the windowed-guidance runs showed layout is set
RANK = 3  # dimensions of each class's shared subspace
LOW_CUTOFF = 4  # low-pass radius for freq, in cycles per 32-pixel latent (Nyquist is 16)
FEATURE_BLOCKS = list(range(13, 24))  # the later blocks, where w = 4 lowered effective rank in the 16-seed run
TOKEN_BLOCKS = list(range(8, 16))  # tfeat's blocks: where cond - uncond is most shared across seeds (cond_tokens.py)
TOKEN_RANK = 16
CADS_LO, CADS_HI = 0.1, 0.4  # cads: pure noise conditioning before CADS_LO, annealed to the clean embedding by CADS_HI
MOD_CHUNKS = ["shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"]  # adaLN_modulation output order
BLOCK_GROUPS = [range(0, 8), range(8, 16), range(16, 24)]  # bcond's groups; the final layer has its own strength
# The conditions the report's intervention table uses (score_0.5 also appears in its shuffled-push comparison).
CONDITIONS = ["w1", "w4", "scalar_1", "score_0.5", "score_1", "freq_1", "cond_0.5", "cond_1", "feat_1", "wrongclass_1"]


def build_tables(run, path, seeds=range(16)):
    """Rank-3 bases per class from the mean Delta (latent) and mean cond-uncond difference (features) over early steps.

    Means are taken over seeds, so seed-specific parts cancel and what remains is the shared component. An intervened trajectory lies
    between w=1 and w=4, and the shared direction differs between them later in the window, so both are included.
    """
    run = Path(run)
    latents, features = [], []
    for w in ("w1", "w4"):
        files = [np.load(run / w / f"seed_{s}" / "trajectory.npz") for s in seeds]
        s_t = files[0]["s_t"]
        early = s_t < EARLY_END  # boolean mask over steps: True for the steps inside the intervention window
        # (seeds, early steps, 8 classes, 3 guided channels, 32, 32). float64 for the averaging and SVD below.
        delta = np.stack([f["delta"][early][:, :, :GUIDED_CHANNELS] for f in files], 0).astype(np.float64)
        latents.append(delta.mean(0).reshape(early.sum(), 8, -1))  # (steps, classes, 3072)
        # Features: the class signal inside the network, pooled conditional minus pooled unconditional activation per block.
        pooled = [np.load(run / w / f"seed_{s}" / "pooled.npy", mmap_mode="r") for s in seeds]
        diff = np.mean([np.asarray(p[early][:, FEATURE_BLOCKS, 0] - p[early][:, FEATURE_BLOCKS, 1], dtype=np.float64) for p in pooled], 0)
        features.append(diff)  # (steps, blocks, classes, 1024)
    # Stack the w=1 and w=4 steps as rows, so each class gets one (2 * early steps, D) matrix of mean directions.
    latents, features = np.concatenate(latents, 0), np.concatenate(features, 0)
    # The top RANK right singular vectors span the directions those mean vectors mostly lie in. They are orthonormal rows, so the
    # projection of any vector v onto their span is basis.T @ (basis @ v). The matrix is not centered on purpose: the mean
    # direction itself is the shared push to capture, not variation around it.
    top = lambda x: np.linalg.svd(x, full_matrices=False)[2][:RANK]
    latent_basis = np.stack([top(latents[:, c]) for c in range(8)]).astype(np.float32)  # (8, 3, 3072), orthonormal rows
    feature_basis = np.stack([np.stack([top(features[:, b, c]) for c in range(8)]) for b in range(len(FEATURE_BLOCKS))]).astype(np.float32)  # (11 blocks, 8, 3, 1024)
    np.savez(path, latent_basis=latent_basis, feature_basis=feature_basis, seeds=np.array(list(seeds)), early_end=EARLY_END, rank=RANK)


def build_token_tables(run, path):
    """tfeat's bases from a guidance_capture.py --token-steps run (w = 4): per block in TOKEN_BLOCKS and class, the top
    TOKEN_RANK right singular vectors of the seed-averaged token push d, stacked over the early token steps.

    As in build_tables, averaging over seeds keeps the shared part, and the matrix is not centered: the mean push is the target.
    """
    run = Path(run)
    folder = run / "w4"
    seeds = sorted(int(p.parent.name[5:]) for p in folder.glob("seed_*/metadata.json"))
    steps = json.loads((run / "manifest.json").read_text())["token_steps"]
    s_t = np.load(folder / f"seed_{seeds[0]}" / "trajectory.npz")["s_t"]
    early = [i for i, k in enumerate(steps) if s_t[k] < EARLY_END]
    D = [np.load(folder / f"seed_{s}" / "tokens_delta.npy", mmap_mode="r") for s in seeds]
    n_classes = D[0].shape[2]
    basis = np.zeros((len(TOKEN_BLOCKS), n_classes, TOKEN_RANK, D[0].shape[-1]), np.float32)
    for j, b in enumerate(TOKEN_BLOCKS):
        for c in range(n_classes):
            # (early steps * 256 tokens, 1024): the shared push at every token and early step.
            mean = np.concatenate([np.mean([x[i, b, c].astype(np.float64) for x in D], 0) for i in early], 0)
            basis[j, c] = np.linalg.svd(mean, full_matrices=False)[2][:TOKEN_RANK]
    np.savez(path, token_basis=basis, seeds=np.array(seeds), blocks=np.array(TOKEN_BLOCKS), steps=np.array([steps[i] for i in early]),
             early_end=EARLY_END, rank=TOKEN_RANK)


def low_pass_matrix():
    """Real 1024 x 1024 matrix for the radial low-pass filter, so it can run as one matmul on any device."""
    f = np.fft.fftfreq(32, 1 / 32)  # integer frequencies -16..15, in cycles per 32-pixel image
    mask = np.hypot(*np.meshgrid(f, f, indexing="ij")) <= LOW_CUTOFF  # keep frequencies within radius LOW_CUTOFF of zero
    # Filter each of the 1024 one-pixel images. Row i is then the filter's response to pixel i, and since the filter is linear,
    # filtering any image x equals x_flat @ M. The mask is symmetric, so the response is real (.real drops rounding-level imaginary
    # parts) and M is symmetric, which is why callers can use M.T and M interchangeably.
    eye = np.eye(1024).reshape(1024, 32, 32)
    return np.fft.ifft2(np.fft.fft2(eye) * mask).real.reshape(1024, 1024).astype(np.float32)


def parse(condition):
    """'score_0.5' -> ('score', 0.5); 'bcond_1-0-0.5' -> ('bcond', (1.0, 0.0, 0.5)). 'w1' and 'w4' have no strength: 0."""
    if condition in ("w1", "w4"):
        return condition, 0.0
    method, strength = condition.split("_")
    if method == "bcond":
        strengths = tuple(float(s) for s in strength.split("-"))
        assert len(strengths) in (len(BLOCK_GROUPS), len(BLOCK_GROUPS) + 1), f"bcond needs one strength per block group: {condition}"
        assert all(0 <= x <= W / (W - 1) + 1e-9 for x in strengths), f"bcond strengths must give 0 <= a <= 1: {condition}"
        return method, strengths + strengths[-1:] * (len(strengths) == len(BLOCK_GROUPS))  # final layer defaults to the last group
    return method, float(strength)


class Intervention:
    """score and cond interventions in a form guidance_capture.py can drive. A spec joins parts with '+', e.g. 'score_1+cond_0.5'.

    The arithmetic matches run_condition exactly, so a single-part spec reproduces its samples bit for bit. In a mix, cond changes
    the embedding first, so score projects the Delta that the damped embedding produced.

    guidance_capture.py calls these methods in this order:
      install(model)    once before sampling: registers the embedding hook if the spec includes cond
      set_time(t)       at each step before the model runs, so hooks know whether the step is inside the early window
      adjust(out, last) at each step after forward_with_cfg, to apply the score part to the guided velocity
      remove()          once after sampling: unregisters the hooks
    """

    def __init__(self, spec, tables, n):
        self.parts = dict(parse(part) for part in spec.split("+"))  # e.g. {"score": 1.0, "cond": 0.5}
        assert set(self.parts) <= {"score", "cond"}, f"unsupported intervention: {spec}"
        self.tables, self.n, self.active, self.handles = tables, n, False, []

    def set_time(self, t):
        self.active = t < EARLY_END

    def install(self, model):
        if "cond" in self.parts:
            alpha = 1 - self.parts["cond"] * (W - 1) / W  # at strength 1: 1 - 3/4 = 1/4 = 1/W
            n = self.n

            def embed_hook(_module, _inputs, emb):
                # emb is the label embedding for the whole doubled batch: rows [:n] real classes, rows [n:] the null class.
                if not self.active:
                    return None  # outside the window: leave the embedding unchanged
                # Move each class embedding a fraction alpha of the way from the null embedding. SiT adds this to the time
                # embedding and every block reads the sum, so the class signal is weakened everywhere in the network at once.
                return torch.cat([emb[n:] + alpha * (emb[:n] - emb[n:]), emb[n:]], 0)
            self.handles.append(model.y_embedder.register_forward_hook(embed_hook))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def adjust(self, out, last):
        """out: forward_with_cfg's result; last: the raw branch outputs it was built from. Returns the adjusted out."""
        if not (self.active and "score" in self.parts):
            return out
        n = self.n
        delta = (last[:n, :GUIDED_CHANNELS] - last[n:, :GUIDED_CHANNELS]).float()  # (n, 3, 32, 32)
        basis = self.tables["latent"]  # (n classes, 3, 3072): sample i uses its own class's basis
        flat = delta.reshape(n, -1)
        # Projection P Delta = basis.T @ (basis @ Delta), done per sample: first the 3 coordinates (n, 3), then back to (n, 3072).
        part = torch.einsum("nk,nkd->nd", torch.einsum("nkd,nd->nk", basis, flat), basis).reshape(delta.shape)
        out = out.clone()  # do not edit forward_with_cfg's tensor in place
        # out is f_cond + (W-1) Delta on the guided channels; removing (W-1) * L * P Delta leaves f_cond + (W-1)(Delta - L P Delta).
        out[:n, :GUIDED_CHANNELS] -= (W - 1) * self.parts["score"] * part
        return out


def load_tables(path, device):
    """Load build_tables output onto the device, plus the low-pass matrix. Returns (tables, calibration seeds)."""
    saved = np.load(path)
    return dict(latent=torch.from_numpy(saved["latent_basis"]).to(device), feature=torch.from_numpy(saved["feature_basis"]).to(device),
                low_pass=torch.from_numpy(low_pass_matrix()).to(device)), saved["seeds"].tolist()


def run_condition(model, sample_fn, condition, seed, device, tables):
    """Sample one batch under one condition. Returns (initial noise, final latents).

    A condition may join one score-space part (score, wrongclass, freq), one token part (tfeat, tfeatwrong) and one late_<w>
    part with '+', e.g. 'tfeat_1.3333+score_1+late_5'. They act at different places (inside the blocks, on the returned
    velocity, and on the guidance scale after the window), so they compose.
    """
    parts = dict(parse(part) for part in condition.split("+"))
    LATENT, TOKEN = ("score", "wrongclass", "freq"), ("tfeat", "tfeatwrong", "tshuffle")
    EMBED = ("cond", "cads", "cnoise")  # act on the class embedding
    MOD = ("mcond", "mchunks", "mfrom", "mto")  # mcond_L damps chosen adaLN modulation chunks (bitmask mchunks) in blocks mfrom..mto
    TOKEN_MODIFIERS = ("tfeatlate", "tend", "trank", "tfrom")  # change how the token part acts; need tfeat or tfeatwrong
    SLG = ("slg", "slgfrom", "slgto")  # skip-layer guidance: needs the plain forward, so no hook-based parts with it
    assert not ("slg" in parts and set(parts) & {*TOKEN, *EMBED, *MOD}), f"slg cannot be combined with hook-based parts: {condition}"
    assert len(parts) == 1 or (set(parts) <= {*LATENT, *TOKEN, *TOKEN_MODIFIERS, *EMBED, *MOD, *SLG, "late", "smooth", "wall"}
                               and sum(m in LATENT for m in parts) <= 1 and sum(m in TOKEN for m in parts) <= 1), \
        f"only one score-space part, one token part, cond, late and smooth can be combined: {condition}"
    # smooth_<d>: replace the step at EARLY_END by a cosine ramp over EARLY_END +- d. Every damping strength is scaled by the
    # gate g(t), which falls from 1 to 0 across the ramp, and the guidance scale moves from W to the late value as
    # W g + w_late (1 - g). The ramp is symmetric about EARLY_END, so the total damping over time is unchanged.
    ramp = parts.get("smooth")
    assert ramp is None or not {"tend", "tfeatlate"} & set(parts), f"smooth replaces tend and tfeatlate: {condition}"

    def gate(t):
        if ramp is None:
            return 1.0 if t < EARLY_END else 0.0  # the plain step
        x = (t - (EARLY_END - ramp)) / (2 * ramp)  # 0 where the ramp starts, 1 where it ends
        return 1.0 if x <= 0 else 0.0 if x >= 1 else 0.5 * (1 + math.cos(math.pi * x))
    method, strength = next(iter(parts.items())) if len(parts) == 1 else (None, None)  # single-method branches below
    latent_method = next((m for m in LATENT if m in parts), None)
    token_method = next((m for m in TOKEN if m in parts), None)
    assert token_method or not set(parts) & set(TOKEN_MODIFIERS), f"token modifiers need tfeat or tfeatwrong: {condition}"
    token_end = parts.get("tend", EARLY_END)  # tend_<t>: the token part acts only while s_t < t (score keeps EARLY_END)
    assert token_end <= EARLY_END, "tend cannot extend the window; use tfeatlate for steps after it"
    token_rank = int(parts.get("trank", TOKEN_RANK))  # trank_<k>: only the top k basis directions
    token_blocks = [b for b in TOKEN_BLOCKS if b >= parts.get("tfrom", 0)]  # tfrom_<b>: only blocks b and later
    n = len(LABELS)
    z = initial_noise(seed)
    x = torch.cat([z, z], 0).to(device)
    y = torch.tensor(LABELS + [NULL_CLASS] * n, device=device)
    # State shared between the sampler wrapper and the hooks: whether this step is in the window, and the raw branch outputs.
    state = dict(active=False, last=None, t=0.0, g=0.0)
    # Seed-determined randomness for cads, cnoise and tshuffle, so every condition is reproducible from its seed alone.
    noise_gen = torch.Generator().manual_seed(seed + 7919)
    if token_method == "tshuffle":
        # One random rotation per sample inside its class's token subspace, fixed over steps, tokens and blocks.
        q, r = torch.linalg.qr(torch.randn(n, token_rank, token_rank, generator=torch.Generator().manual_seed(seed + 104729)))
        rotation = (q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[:, None, :]).to(device)  # Haar-distributed
    excess = (W - 1) / W  # the share of the class signal that cond and feat remove at strength 1

    basis = None
    if latent_method in ("score", "wrongclass"):
        # wrongclass: roll(-1) shifts the class axis, so sample i gets class i+1's subspace (wrapping around). Same rank and
        # same kind of direction, but the wrong class. If this damps as well as score, class-specific directions do not matter.
        basis = tables["latent"] if latent_method == "score" else tables["latent"].roll(-1, 0)
    if token_method:
        # tfeatwrong: the same class shift for the token bases, whose class axis is 1: (blocks, classes, rank, 1024).
        token_basis = tables["token"].roll(-1, 1) if token_method == "tfeatwrong" else tables["token"]

    forward = model.forward

    def recording_forward(xb, tb, yb):
        # Only keeps the raw branch outputs, so the score-space methods can rebuild Delta after forward_with_cfg mixes them.
        state["last"] = forward(xb, tb, yb)
        return state["last"]

    def guided(xb, tb, **kwargs):
        # The function the sampler integrates. It sets the window flag first, because the hooks inside the forward pass read it.
        state["t"] = float(tb[0])
        state["g"] = gate(state["t"])
        state["active"] = state["g"] > 0  # without smooth: exactly s_t < EARLY_END
        if "wall" in parts:
            kwargs["cfg_scale"] = parts["wall"]  # wall_<w>: guidance scale w at every step (late_<w> still overrides after)
        if "late" in parts and state["g"] < 1:
            # late_<w>: guidance scale w after the window, W before (g = 0 gives exactly w, as with the plain step).
            kwargs["cfg_scale"] = W * state["g"] + parts["late"] * (1 - state["g"])
        if method == "w1" or (method == "scalar" and state["active"]):
            # w1: no guidance at any step. scalar: a smaller w during the window, the same in every direction.
            kwargs["cfg_scale"] = 1.0 if method == "w1" else 1 + (W - 1) * (1 - strength)
        out = model.forward_with_cfg(xb, tb, **kwargs)
        if state["active"] and latent_method:
            last = state["last"]
            delta = (last[:n, :GUIDED_CHANNELS] - last[n:, :GUIDED_CHANNELS]).float()
            if latent_method == "freq":
                # Low-pass each channel's 32 x 32 map with one matmul: (n, 3, 1024) @ (1024, 1024).
                part = (delta.reshape(n, GUIDED_CHANNELS, 1024) @ tables["low_pass"].T).reshape(delta.shape)
            else:
                # Projection onto the per-class subspace, as in Intervention.adjust.
                flat = delta.reshape(n, -1)
                part = torch.einsum("nk,nkd->nd", torch.einsum("nkd,nd->nk", basis, flat), basis).reshape(delta.shape)
            out = out.clone()
            out[:n, :GUIDED_CHANNELS] -= (W - 1) * (parts[latent_method] * state["g"]) * part  # excess guidance loses L g * part
        if "slg" in parts:
            # Skip-layer guidance: push away from the conditional prediction of a degraded model (blocks slgfrom..slgto
            # bypassed), f += s (f_cond - f_cond_skip). Class-agnostic quality guidance; costs one extra pass on n rows.
            lo, hi = int(parts.get("slgfrom", 10)), int(parts.get("slgto", 12))
            skips = [model.blocks[b].register_forward_hook(lambda _m, inputs, _o: inputs[0]) for b in range(lo, hi + 1)]
            try:
                f_skip = forward(xb[:n], tb[:n], kwargs["y"][:n])
            finally:
                for h in skips:
                    h.remove()
            out = out.clone()
            out[:n, :GUIDED_CHANNELS] += parts["slg"] * (state["last"][:n, :GUIDED_CHANNELS] - f_skip[:, :GUIDED_CHANNELS])
        return out

    def embed_hook(_module, _inputs, emb):  # rows [:n] are class embeddings, rows [n:] the null embedding
        y, null, changed = emb[:n], emb[n:], False
        if "cond" in parts and state["active"]:
            # cond: same arithmetic as Intervention.install, so the two code paths give bit-identical samples.
            y, changed = null + (1 - parts["cond"] * state["g"] * excess) * (y - null), True
        if "cads" in parts and state["t"] < CADS_HI:
            # CADS (Sadat et al. 2024): y' = sqrt(gamma) y + s sqrt(1 - gamma) eps, then rescaled to y's per-sample mean and
            # std (psi = 1). gamma rises from 0 at CADS_LO to 1 at CADS_HI; fresh noise every step. eps is scaled by y's std so
            # that s is relative to the embedding's own size.
            gamma = min(1.0, max(0.0, (state["t"] - CADS_LO) / (CADS_HI - CADS_LO)))
            eps = torch.randn(y.shape, generator=noise_gen).to(y.device)
            mu, sd = y.mean(1, keepdim=True), y.std(1, keepdim=True)
            mixed = gamma ** 0.5 * y + parts["cads"] * (1 - gamma) ** 0.5 * sd * eps
            y, changed = (mixed - mixed.mean(1, keepdim=True)) / mixed.std(1, keepdim=True) * sd + mu, True
        if "cnoise" in parts and state["active"]:
            # cnoise: a fresh random offset to the class embedding at every step of the window, s times its std; the class
            # signal itself is kept at full strength.
            eps = torch.randn(y.shape, generator=noise_gen).to(y.device)
            y, changed = y + parts["cnoise"] * state["g"] * y.std(1, keepdim=True) * eps, True
        return torch.cat([y, null], 0) if changed else None

    def modulation_hook(_module, _inputs, mod):
        # mcond: mod = adaLN_modulation(c), (2n, 6 * width): the only route by which the class reaches a block. Selected
        # chunks of the conditional rows move toward the null rows' values, as cond does to the embedding.
        if not state["active"]:
            return None
        width = mod.shape[1] // len(MOD_CHUNKS)
        alpha = 1 - parts["mcond"] * state["g"] * excess
        cond = mod[:n].clone()
        for j in range(len(MOD_CHUNKS)):
            if int(parts.get("mchunks", 63)) >> j & 1:
                sl = slice(j * width, (j + 1) * width)
                cond[:, sl] = mod[n:, sl] + alpha * (mod[:n, sl] - mod[n:, sl])
        return torch.cat([cond, mod[n:]], 0)

    def block_hook(index):
        # feat: index is the position in FEATURE_BLOCKS (0-10), which is how feature_basis is indexed, not the block number.
        def hook(_module, _inputs, out):
            if not state["active"]:
                return None
            u = tables["feature"][index]  # (8, 3, 1024)
            # The block's class signal: pooled conditional minus pooled unconditional output, one 1024-d vector per class.
            diff = out[:n].mean(1) - out[n:].mean(1)
            part = torch.einsum("nk,nkd->nd", torch.einsum("nkd,nd->nk", u, diff), u)  # its projection onto the class's subspace
            return torch.cat([out[:n] - excess * strength * part[:, None, :], out[n:]], 0)  # same shift on every token
        return hook

    def condition_hook(alpha):
        # bcond: a pre-hook on one block (or the final layer), whose forward takes (x, c). c = t + y and t is the same in both
        # halves, so interpolating c between its null and class rows is the same as interpolating y, for this block alone.
        def hook(_module, inputs):
            if not state["active"]:
                return None
            x, c = inputs
            return x, torch.cat([c[n:] + alpha * (c[:n] - c[n:]), c[n:]], 0)
        return hook

    def token_hook(index):
        # tfeat: index is the position in TOKEN_BLOCKS. Each conditional token's push d = cond - uncond is projected onto its
        # class's subspace, and that part is reduced, the same shift a feat edit makes but separately at every token.
        def hook(_module, _inputs, out):
            # Inside the token window: the token part's strength. After the early window: tfeatlate's strength, if given, so
            # that a late guidance boost cannot push the class-shared directions back. In between (tend < t < EARLY_END): off.
            if ramp is not None:
                if state["g"] == 0:
                    return None
                s = parts[token_method] * state["g"]
            elif state["t"] < token_end:
                s = parts[token_method]
            elif not state["active"] and "tfeatlate" in parts:
                s = parts["tfeatlate"]
            else:
                return None
            u = token_basis[index][:, :token_rank]  # (8 classes, rank, 1024); rows are sorted by singular value
            d = out[:n] - out[n:]  # (n, tokens, 1024)
            coords = torch.einsum("ntd,nkd->ntk", d, u)
            part = torch.einsum("ntk,nkd->ntd", coords, u)
            if token_method == "tshuffle":
                # Replace the subspace part by its rotated version instead of removing it: same size, seed-specific direction.
                part = part - torch.einsum("ntk,nkd->ntd", torch.einsum("ntk,njk->ntj", coords, rotation), u)
            return torch.cat([out[:n] - excess * s * part, out[n:]], 0)
        return hook

    # Register only the hooks the chosen method needs; the score-space methods need none.
    handles = []
    if set(parts) & set(EMBED):
        handles.append(model.y_embedder.register_forward_hook(embed_hook))
    if "mcond" in parts:
        lo, hi = int(parts.get("mfrom", 0)), int(parts.get("mto", len(model.blocks) - 1))
        handles += [model.blocks[b].adaLN_modulation.register_forward_hook(modulation_hook) for b in range(lo, hi + 1)]
    if method == "bcond":
        # strength holds one value per block group, then one for the final layer.
        for modules, s in zip([[model.blocks[b] for b in g] for g in BLOCK_GROUPS] + [[model.final_layer]], strength):
            if s == 0:
                continue  # alpha = 1 leaves c unchanged; skipping the hook keeps those modules bit-identical to w4
            handles += [m.register_forward_pre_hook(condition_hook(1 - s * excess)) for m in modules]
    if token_method:
        handles += [model.blocks[b].register_forward_hook(token_hook(i)) for i, b in enumerate(TOKEN_BLOCKS) if b in token_blocks]
    if method == "feat":
        handles += [model.blocks[b].register_forward_hook(block_hook(i)) for i, b in enumerate(FEATURE_BLOCKS)]
    model.forward = recording_forward
    try:
        # Hooks read state["active"], which guided sets before the model runs.
        final = sample_fn(x, guided, y=y, cfg_scale=W)[-1][:n].float().cpu().numpy()
    finally:
        del model.__dict__["forward"]
        for handle in handles:
            handle.remove()
    assert np.isfinite(final).all(), f"non-finite final latents: {condition} seed {seed}"
    return z.numpy(), final


def main():
    """Either fit the tables (--build-tables), or sample every (seed, condition) pair that is not already saved."""
    p = argparse.ArgumentParser()
    p.add_argument("--build-tables", action="store_true")
    p.add_argument("--calibration")
    p.add_argument("--tables")
    p.add_argument("--weights")
    p.add_argument("--out")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--device", default="mps")
    p.add_argument("--token-tables", help="tfeat bases; with --build-token-tables, where to write them")
    p.add_argument("--build-token-tables", action="store_true", help="fit tfeat bases from the --calibration token run")
    a = p.parse_args()
    if a.build_tables:
        build_tables(a.calibration, a.tables)
        return
    if a.build_token_tables:
        build_token_tables(a.calibration, a.token_tables)
        return

    torch.set_grad_enabled(False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    saved = np.load(a.tables)
    # Evaluation seeds must differ from the seeds the tables were fit on. The one exception is a folder named "calibration",
    # which reruns calibration seeds on purpose to check that this script reproduces the capture run bit for bit.
    assert not set(a.seeds) & set(saved["seeds"].tolist()) or "calibration" in out.name, "evaluation seeds overlap the calibration seeds"
    tables = dict(latent=torch.from_numpy(saved["latent_basis"]).to(a.device), feature=torch.from_numpy(saved["feature_basis"]).to(a.device),
                  low_pass=torch.from_numpy(low_pass_matrix()).to(a.device))
    manifest = dict(weights=a.weights, weights_sha256=sha256(a.weights), tables=a.tables, tables_sha256=sha256(a.tables), w=W, early_end=EARLY_END,
                    rank=RANK, low_cutoff=LOW_CUTOFF, feature_blocks=FEATURE_BLOCKS, grid_points=a.steps, model_calls=a.steps - 1,
                    branch_evaluations=2 * (a.steps - 1), sampler="euler", guided_channels=GUIDED_CHANNELS, classes=LABELS, device=a.device,
                    calibration_seeds=saved["seeds"].tolist())
    if a.token_tables:  # only runs that use tfeat carry these keys, so older manifests still match
        token = np.load(a.token_tables)
        assert not set(a.seeds) & set(token["seeds"].tolist()), "evaluation seeds overlap the token-table calibration seeds"
        tables["token"] = torch.from_numpy(token["token_basis"]).to(a.device)
        manifest.update(token_tables=a.token_tables, token_tables_sha256=sha256(a.token_tables), token_blocks=TOKEN_BLOCKS, token_rank=TOKEN_RANK)
    path = out / "manifest.json"
    if path.exists():  # resuming: settings must match the samples already saved
        assert json.loads(path.read_text()) == manifest, "settings differ from the existing manifest in --out"
    else:
        path.write_text(json.dumps(manifest, indent=2))

    model = load_model("SiT-L/2", a.weights, a.device)
    sample_fn = Sampler(create_transport("Linear", "velocity", None, None, None)).sample_ode(
        sampling_method="euler", num_steps=a.steps, atol=1e-6, rtol=1e-3, reverse=False)
    for seed in a.seeds:  # seeds are the outer loop, so stopping early leaves every condition on the same seeds
        for condition in a.conditions:
            target = out / condition / f"seed_{seed}.npz"
            if target.exists():  # finished in an earlier invocation
                continue
            target.parent.mkdir(exist_ok=True)
            start = time.perf_counter()
            z, final = run_condition(model, sample_fn, condition, seed, a.device, tables)
            seconds = time.perf_counter() - start
            tmp = target.with_suffix(".tmp.npz")
            np.savez(tmp, seed=seed, z=z, final=final, seconds=seconds, noise_sha256=array_sha256(z), final_sha256=array_sha256(final))
            tmp.rename(target)  # a file at the final path is always complete
            print(f"{condition} seed={seed}: {seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
