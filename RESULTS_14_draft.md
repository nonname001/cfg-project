## 14. Where diversity is lost: layout early, semantics throughout

> Draft for review (2026-11-27). Not yet merged into RESULTS.md. The full timestamped log is `outputs/guidance_delta/night_notes.md` (sessions 4-6).

**Question.** Section 12 found no small circuit and no intervention clearly better than condition damping. This section goes back
to the project's main question, *where in sampling and in the network is diversity lost*, with measurements designed to locate it
rather than to fix it.

**Short version.** Guidance removes two kinds of diversity, through two parts of the guidance vector, at different times.
**Layout** diversity is lost only before about t = 0.2, to the part of Δ that every seed of a class shares, which costs little
fidelity; score damping removes exactly this part. **Semantic** diversity is lost to guidance at any time up to about t = 0.6, and
from t ≈ 0.24 on it is carried by the seed-specific part of Δ, the same part that carries most of the early fidelity gain. That
coupling is why every per-sample intervention in section 12 lands on one diversity-fidelity curve. Inside the network the effect is
spread over blocks, heads and time; it is readable in blocks 15-16 by t ≈ 0.16, but no block, head, attention pattern or extra
feature direction carries it on its own.

### 14.1 Late guidance repairs samples and removes real diversity

**Setup.** Guidance only before t = 0.3 (`late_1`: w = 4, then w = 1) and only after (`scalar_1`: w = 1, then w = 4), seeds 16-47,
against the 1,300 real training images per class (`bench/window_fidelity.py`). Section 9 left open whether the late window's loss of
DINO/CLIP diversity was real or a quality gain.

| Seeds 16-47 | w = 1 | early-only | late-only | w = 4 | real images |
|---|---|---|---|---|---|
| Precision | 0.26 | 0.34 | 0.51 | 0.73 | |
| DINO diversity | 0.655 | 0.370 | 0.370 | 0.244 | 0.494 |
| Distance to the real class mean (typicality) | 0.50 | 0.255 | 0.235 | 0.175 | 0.30 |
| Coverage (Naeem et al.) | 0.008 | 0.016 | 0.032 | 0.053 | |

- **Late guidance carries most of the fidelity gain.** Late-only keeps 0.54 [0.43, 0.64] of the precision gain, early-only 0.18.
- **Each window alone removes 0.69 of the semantic diversity loss**, so the two halves of guidance are not "layout early, quality
  late": both remove semantic diversity.
- **It is real diversity loss, not only repair.** Among samples inside the real manifold under both w = 1 and late-only, late
  guidance still lowers DINO diversity from 0.62 to 0.38 (change −0.24 [−0.33, −0.16]).
- **w = 1 is more varied than real photos and w = 4 more typical than them.** Real images sit between the two on both measures.
  Coverage rises with guidance because most w = 1 samples lie off the real manifold.

### 14.2 When each sample commits

**Setup.** From the saved 16-seed trajectories, switch the guidance scale at step k and finish sampling (`bench/switch_sweep.py`;
restarting from a saved state reproduces the saved samples exactly). For each sample, the final image is placed on the line from
that seed's w = 1 image (0) to its w = 4 image (1), in DINO space (semantics) and in the low band of the latents (layout).

| Switch at s_t | 0.06 | 0.12 | 0.20 | 0.31 | 0.41 | 0.57 |
|---|---|---|---|---|---|---|
| w = 4 before, w = 1 after: layout | 0.34 | 0.62 | 0.86 | 0.95 | 0.98 | 0.99 |
| w = 4 before, w = 1 after: semantics | 0.36 | 0.59 | 0.76 | 0.86 | 0.92 | 0.97 |
| w = 1 before, w = 4 after: layout | 0.67 | 0.33 | 0.15 | 0.05 | 0.02 | 0.01 |
| w = 1 before, w = 4 after: semantics | 0.89 | 0.84 | 0.74 | 0.65 | 0.56 | 0.35 |

(Seeds 0-15, 128 samples; medians over samples, 95% intervals within ±0.03.)

- **Layout is decided by about t = 0.2** (half-committed at a median t = 0.10, IQR 0.07-0.12) and later guidance barely changes it.
- **Semantics stay open.** They start moving as early (half-committed at a median t = 0.09), but guidance that starts only at
  t = 0.31 still moves a sample 0.65 [0.63, 0.66] of the way to its w = 4 semantics, from t = 0.57 still 0.35 [0.34, 0.36].
- **Guidance from t = 0.12 on** loses almost all semantic diversity (DINO 0.269, w = 4 0.228) but keeps most layout diversity
  (layout spread 0.760, w = 1 0.819), at w = 4's precision (0.71 vs 0.68): the guidance-interval effect, seen per sample.
- The hybrids lie far off the line between a seed's w = 1 and w = 4 images in DINO space (0.4-0.6 of its length): guidance at
  different times reaches different semantics, not points in between.

### 14.3 Where the identity is readable inside the network

**Setup.** Ridge probes from each block's pooled output at each step to the final image's top 8 within-class DINO directions,
cross-validated over seeds (`bench/probe_identity.py`; intervals from 12-seed subsets and a permutation null, `bench/probe_ci.py`).

- At w = 4, blocks 15-16 read the final identity from t ≈ 0.16 on (R² 0.16-0.28, permutation null ≤ 0.008), before the decoded
  prediction x̂ shows it (t ≈ 0.3-0.4). Blocks 0-10 and x_t itself never carry it linearly.
- **Correction to an earlier reading:** the difference between w = 4 and w = 1 is not significant at any block or step (for
  example +0.11 [−0.03, +0.31] at t = 0.16, block 15), so the probes do *not* show that guidance makes samples commit earlier.

### 14.4 Patching between the w = 1 and w = 4 runs of a seed

> **Clarification: why not plain activation patching?** At a given step the two runs evaluate the same network with the same time
> and class embeddings; guidance only mixes the two branch outputs afterwards. They differ only through x_t. Replacing a block's
> full output (the residual stream) with the other run's makes every later block, and the output, exactly the other run's,
> whichever block is chosen. So the experiment patches parts instead.

**Setup** (`bench/patch_sweep.py`, `bench/patch_analysis.py`; seeds 0-7, 64 samples per cell; the shared / seed-specific split
on seeds 0-15, 128 samples). Three kinds of patch, from the
donor run into the recipient run, at t = 0.10, 0.24 or 0.41:

| Patch | What it replaces |
|---|---|
| block write | block b's own attention + MLP contribution, for 5 steps |
| velocity transplant | the whole network output for 5 steps (the most any 5-step intervention there can transfer) |
| x_t band | one frequency band of the state x_t (low ≤ 4 cycles, mid, high), once |
| low band, shared / seed-specific | only the class-mean part of the low band (over the other 15 seeds), or only the seed's deviation from it |

With the donor set to the recipient, every patch reproduces the saved samples (to float rounding).

Results, moving w = 4's outcome into the w = 1 run (median movement toward the donor's outcome, 0 to 1):

| | t = 0.10 | t = 0.24 | t = 0.41 |
|---|---|---|---|
| Low band of x_t: semantics / layout | 0.51 / 0.54 | **0.72 / 0.86** | 0.68 / 0.91 |
| Low band, shared part: semantics | **0.41** [0.33, 0.47] | 0.39 [0.35, 0.46] | 0.22 [0.18, 0.26] |
| Low band, seed-specific part: semantics | 0.18 [0.15, 0.21] | **0.55** [0.48, 0.62] | **0.53** [0.47, 0.56] |
| Mid / high band: semantics | | 0.15 / 0.06 | 0.15 / 0.22 |
| Velocity transplant, 5 steps: semantics | 0.27 | 0.39 | 0.27 |
| Any single block write: semantics | ≤ 0.05 | ≤ 0.10 | ≤ 0.07 |

- **The guided outcome sits mostly in the coarse layout of the state.** By t = 0.24 the low band alone carries most of what a full
  state swap carries (about 0.83, from 14.2); finer bands start to matter only by t = 0.41.
- **The shared and seed-specific parts cross over at about t ≈ 0.2**, when layout finishes committing (14.2). Early (t = 0.10)
  the shared class push carries the guided outcome, semantics included; from t ≈ 0.24 on, the seed-specific part carries the
  semantics. Two extra switch times (seeds 0-6 only, sampling stopped at the deadline) place the handover:

  | Semantics, shared / seed-specific | t = 0.10 | 0.16 | 0.24 | 0.33 | 0.41 |
  |---|---|---|---|---|---|
  | w = 4's outcome into the w = 1 run | 0.41 / 0.18 | 0.43 / 0.36 | 0.39 / 0.55 | 0.25 / 0.55 | 0.22 / 0.53 |
  | w = 1's outcome into the w = 4 run | 0.09 / 0.04 | 0.11 / 0.08 | 0.06 / 0.11 | 0.06 / 0.13 | 0.04 / 0.10 |

  (t = 0.16 and 0.33: 7 seeds, 56 samples; the others 16 seeds.)
- **Undoing with the seed-specific part costs fidelity**: precision falls from 0.68 to 0.58 at t = 0.24 and 0.52 at t = 0.41, for
  DINO recovery of +0.07 and +0.12. Undoing with the shared part keeps precision (0.69-0.70) and recovers almost no diversity
  (+0.03 to +0.05).
- **Inside the network the effect is spread out.** No single block's write moves semantics by more than 0.10. As a share of the
  5-step velocity transplant, block 15 carries the most when undoing (0.30-0.41); when imposing, blocks 10-20 are about equal.
- **Undoing is harder than imposing** at every cell, and a 5-step window moves semantics only 0.1-0.4: the effect is spread over
  time.

### 14.5 Which part of early guidance carries fidelity

**Setup.** Two new score-space methods in `bench/guidance_intervene.py`, using the same rank-3 shared subspace P_c as score damping,
during t < 0.3: `spec_L` removes the seed-specific part (I − P_c)Δ of the excess guidance (the converse of score damping), and
`swap_L` replaces it with another seed's. Seeds 16-47, 1,300-image reference.

| Seeds 16-47 | Removes | Precision ret. | DINO | Layout vs w = 4 |
|---|---|---|---|---|
| score 1 | shared part | 0.91 | 0.12 | recovered |
| spec 1 | seed-specific part | 0.75 (−0.25 [−0.36, −0.13]) | 0.09 | −0.15 [−0.19, −0.11] |
| swap 1 | seed-specific part, another seed's instead | 0.67 (−0.33 [−0.43, −0.22]) | 0.08 | −0.08 |
| scalar 1 | both | 0.54 | 0.31 | |

- **Early fidelity leans on the seed-specific part** (removing it costs 0.25 of the gain, the shared part 0.09), and that part has
  to match the sample: another seed's is worse than none.
- **The shared part carries the layout collapse; the seed-specific part keeps samples apart.** With only the shared push, layout
  is even less varied than at w = 4.
- This is why score damping sits above the curve (12.7). (On seeds 16-31 alone the split looked even; the pooled 32 seeds reverse
  that, another instance of seed-set variance.)

### 14.6 Explanations that did not hold

| Hypothesis | Test | Result |
|---|---|---|
| Condition damping's extra late-block variation directions (11.2) cause its semantic diversity | ablate those 3 directions after the window, seeds 32-47; control: 3 directions w = 4 also varies along (`bench/extra_dirs.py`) | DINO recovery 0.56 → 0.53 (−0.02 [−0.06, +0.01]), no different from the control; injecting them into w = 4 lands on the curve. Not causal. |
| The network compensates for damping (11.1) | split each change in the class signal into a part along w = 4's signal and a part across it (`bench/along_across.py`) | In the window condition damping is a weakening along the signal (cond 0.5: −0.49 along, 0.21 across); after it every condition is a redirection, mostly seed-specific (score: −0.10 / 0.35). Along stays negative: no compensation. |
| Guidance makes seeds attend to the same places | attention summaries for every block, head and step, seeds 0-7 (`bench/attn_capture.py`, `attn_analysis.py`) | The class changes attention in blocks 8-23 (most in heads of blocks 15-17) about equally at w = 1 and w = 4. In the window, cross-seed similarity of attended positions does not change in blocks 8-23 (+0.004 [−0.000, +0.009], +0.000 [−0.006, +0.007]); it rises slightly in blocks 0-7 and after the window, following the layout convergence. |
| The collapse is what a Gaussian model of each class predicts | exact Gaussian CFG dynamics in the low band, fit to the model's own samples (`bench/gaussian_theory.py`) | Not identifiable from 48 samples per class: the predicted layout ratio ranges 0.64-1.00 over covariance choices. Robust: at w = 1 a linear map from noise predicts 58-75% of each sample's layout; Gaussian w = 4 dynamics over-amplify seed differences; SiT's Δ turns seed-specific faster than a Gaussian's. |

### 14.7 Checks: the VAE and the sampler are not the cause

**VAE** (`bench/vae_check.py`). The 400 real validation images keep their diversity through VAE encode + decode (DINO 0.473 →
0.490, CLIP 0.195 → 0.193; each reconstruction has DINO cosine 0.957 to its original). Guided latents are larger (RMS 0.892 vs
0.764 at w = 1; real images 0.807), but rescaling them before decoding leaves diversity unchanged (w = 4: 0.244 → 0.242; w = 1
scaled up: 0.655 → 0.656). The loss is in SiT's latents.

**Stochastic sampling** (`guidance_intervene.py --sampler sde`: SiT's Euler-Maruyama sampler, 100 steps, the same noise per seed in
every condition; seeds 16-31). Guidance keeps 0.39 of w = 1's DINO diversity with the SDE sampler and 0.38 with the ODE; layout
0.80 vs 0.75. Score damping recovers all lost layout and 0.08 vs 0.10 of the semantic diversity. Stochastic sampling does not
change the picture. (The condition-damping curve under SDE was not run.)

### 14.8 Diversity that counts only realistic images

Mean pairwise distance and Vendi reward spread anywhere in embedding space, including regions no real image occupies: w = 1
scores as more diverse than real photos (DINO 0.655 vs 0.494) although only 26% of its samples fall inside the real manifold.
`bench/realistic_diversity.py` scores every condition against the real images of its class (1,268 per class; 32 more held out as
a "real sample", the target a perfect generator would match): coverage, realistic distinct images per 32 draws (Vendi over the
on-manifold samples only) and KID to the real distribution.

| Seeds 16-47 | Precision | Coverage | Realistic distinct / 32 | KID (×10³) |
|---|---|---|---|---|
| real sample | 0.76 | 0.070 | 7.36 | ≈ 0 |
| w = 1 | 0.26 | 0.007 | 5.40 | 6375 |
| cond 1 | 0.43 | 0.019 | 4.65 | 2805 |
| guidance late only (from t = 0.3) | 0.52 | 0.033 | 4.17 | 2821 |
| score damping | 0.68 | 0.054 | 3.64 | 4710 |
| w = 4 | 0.73 | 0.054 | 3.01 | 6974 |

- Counting only realistic images, w = 4 keeps 0.56 of w = 1's diversity (raw DINO distance: 0.37): about a third of the
  measured loss was off-manifold spread. But w = 4 has 0.41 of real images' realistic diversity at near-real precision, and is
  the poorest match to the real distribution of every condition tested; moderate damping matches it best.
- **On realistic diversity, score damping is no longer above the condition-damping curve.** On seeds 16-47: +0.12 [−0.71, +0.71]
  distinct images per 32 (`bench/realistic_frontier.py`, `figures/frontier_realistic.png`). On 46 seeds (16-61, intervals ±0.4):
  score damping +0.23 [−0.19, +0.61], guidance from t = 0.2 +0.19 [−0.23, +0.58], from t = 0.12 +0.14 [−0.27, +0.51]; the curve is
  monotone at this size (`figures/frontier_realistic_16-61.png`). Its earlier edge (12.7) was layout and raw spread.
  Every method sits far from the real images (precision 0.76, 7.4 distinct per 32).
- **When realistic diversity is lost: the guidance-start family** (`gstart_t` in `guidance_intervene.py`: w = 1 before t, w = 4
  after; seeds 16-47, paired against w = 4):

  | Guidance from t = | Precision (w = 4: 0.73) | Realistic distinct / 32 | Realistic subclusters (K = 8) | vs curve |
  |---|---|---|---|---|
  | 0.06 | 0.73 | +0.27 [+0.04, +0.51] | +0.47 [+0.21, +0.72] | +0.23 (n.s.) |
  | 0.12 | 0.66 | +0.66 [+0.41, +0.92] | +0.63 [+0.36, +0.89] | −0.04 (n.s.) |
  | 0.20 | 0.65 | +0.92 [+0.61, +1.25] | +0.86 [+0.60, +1.10] | +0.14 (n.s.) |

  Only the first ~6% of guidance is free: skipping it keeps w = 4's precision and adds a little realistic diversity. Skipping
  more costs precision at the condition-damping curve's rate. (On seeds 0-15 and 16-31 alone, starting at t = 0.12 looked free;
  pooled, it is not.) Every later start spreads the realistic samples over more real subclusters: the early shared push herds
  them into the dominant one.
- **Subclusters** (`bench/subclusters.py`, `bench/subcluster_commit.py`). The real images of a class form only weak subclusters in
  DINO space (silhouette 0.16-0.44 at K = 2, 0.1-0.17 at K ≥ 3). At K = 8, w = 4 puts half its samples in one subcluster and uses
  about 3 against real images' 6 (on-manifold: 2.9 vs 5.8). Guidance changes the subcluster of 69% of samples; half of that
  choice is made by t ≈ 0.12, but guidance only after t = 0.31 still moves a third of them.

### 14.9 Steering samples to real subclusters (negative so far)

| Method (seeds 16-19, vs w = 4 with the same targets) | Target hit rate | Precision | Realistic diversity |
|---|---|---|---|
| DINO-centroid gradient through the VAE and DINOv2, low-passed, t 0.06-0.3 (`bench/subcluster_steer.py`) | 0.25 vs 0.28 | unchanged | unchanged |
| low band of x_t moved to a real reference image's at t = 0.24, α = 1 (`bench/ref_steer.py`) | 0.38 vs 0.12 | 0.53 vs 0.66 | 1.41 vs 1.60 |

The gradient carries no usable semantics early in sampling, when the decoded prediction is a blur. Reference steering does move
samples toward their targets with little copying (+0.03 DINO cosine to the reference itself), but loses precision and gains no
realistic diversity.

### 14.10 Takeaways

- **Two losses, two mechanisms.** Layout: shared push, before t ≈ 0.2, cheap in fidelity, removable (score damping). Semantics:
  seed-specific guidance, until t ≈ 0.6, and the same component carries the fidelity gain. The second loss cannot be removed per
  sample without losing fidelity, which explains the single frontier of section 12.
- **The decision lives in the state's coarse layout before it lives anywhere in the network.** At t ≈ 0.24 the low band of x_t
  carries most of the guided outcome; inside the network it is distributed, readable in blocks 15-16 but not concentrated there.
- **Measured on realistic images only, the picture is starker.** About a third of the raw semantic "loss" was off-manifold spread
  that w = 1 should not have had. What remains is real: w = 4 has 0.41 of real images' realistic diversity at near-real precision,
  piles half its samples into one real subcluster, and is the poorest match to the real class distribution of every condition
  tested. On this measure every method (damping in score, condition, feature or circuit space; guidance windows and start times;
  lower w) lies on one frontier within noise, robust to the manifold radius (k = 3 and 5), far below the real images. Only the
  first ~6% of guidance is clearly free to drop.
- **Not the VAE, not the sampler.** The decoder neither compresses diversity nor reacts to the larger guided latents, and a
  stochastic sampler loses the same semantic diversity.
- **Limits.** One model (SiT-L/2), w = 4, 8 classes. Switch sweep and patching on the saved 16-seed trajectories (seeds 0-15),
  which are also the seeds the score subspaces were fit on (the patches do not use those subspaces). Precision on 16-64 samples is
  coarse and varies between 16-seed sets; several readings changed when seeds were pooled. Realistic Vendi is noisy (±0.7 at 32
  seeds).

**Open questions this leaves.**
1. Is the seed-specific semantic push itself low-dimensional (a few directions per class that could be steered), or as distributed
   as the class push in section 12?
2. The handover at t ≈ 0.2 coincides with layout commitment. Is it caused by it (once the layout is fixed, only seed-specific
   content is left to decide), or do both follow from the shared push fading (S(Δ) falls to about 0.1 by t = 0.2, section 6.2)?
3. Does the same two-part picture hold for another model (official SiT-XL/2, or a diffusion-trained DiT) and another w?
4. Can samples be sent to under-used real subclusters without losing realism? Two steering attempts (14.9) could not: a feature
   gradient has no grip early in sampling, and replacing the coarse layout steers but costs precision.
