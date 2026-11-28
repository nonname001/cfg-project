# Overnight notes (2026-11-24, 01:05-08:40)

## Summary
- One method beat condition damping on held-out seeds: removing the class dependence of the adaLN *shift* chunks (shift_mlp, or
  shift_mlp + shift_msa) in all blocks during s_t < 0.3 (`mcond_1.3333+mchunks_8` / `mchunks_9`). It came out of the circuit
  analysis. On seeds 24-47 (never used for selection), vs plain w = 4: DINO +0.07 / +0.11, CLIP +0.12 / +0.17, precision retention
  -0.02 / -0.05 (n.s.); vs the condition-damping curve: DINO +0.05 [+0.00,+0.08] / +0.07 [+0.02,+0.11], CLIP +0.08 / +0.09
  (borderline). Modest, and only at the high-fidelity end (precision ret. ~0.95-0.98, DINO recovery ~0.1).
- Everything else landed on or below the curve once held out: CADS, embedding noise, tshuffle, damping shifts+scales
  (screen looked best, failed held-out), gates only, all chunks, stronger or combined variants.
- Mechanism: the class push is spread across blocks and chunks; gates carry most of it (diversity and fidelity together);
  the shift chunks lean towards the class-shared push, which is why damping them adds diversity cheaply.
- Screening on 8 seeds is unreliable: one of tonight's three screen "wins" (shifts+scales) failed on held-out seeds, as did
  every 16-seed dev win before tonight.


Goal: more semantic diversity (DINO/CLIP recovery) at the same real-image fidelity (precision retention) than condition damping.

## Evaluation protocol for tonight
- Every apparent win on 16 dev seeds so far failed on 16 fresh seeds, so: screen new methods on seeds 16-23 (8 seeds, filter only),
  evaluate survivors on 32 pooled seeds 16-47 (`SiT-L_2_night/`, baselines symlinked from `SiT-L_2_tfeat` and `SiT-L_2_confirm`),
  and hold out seeds 48-63 for a final check if time allows (in the end there was no time: 24-47 served as the held-out set).
- Main test: paired bootstrap of recovery minus the condition-damping curve (w4 / cond_0.5 / 0.75 / 1) at the same precision
  retention. Scripts: bench/paired_vs_cond.py, bench/points.py, bench/screen.sh; test outputs in paired_tests/.

## Methods added (guidance_intervene.py)
- `cads_s`: CADS (Sadat et al. 2024) on the class embedding: pure (rescaled) noise before s_t = 0.1, annealed to clean by 0.4.
- `cnoise_s`: class embedding + s * std * fresh noise during the window (class signal kept at full strength).
- `tshuffle_L`: like tfeat, but the mid-block shared-subspace push is rotated per seed within the subspace instead of removed.
- `mcond_L` (+ `mchunks_<bitmask>`, `mfrom_<b>`, `mto_<b>`): damp the class dependence of chosen adaLN modulation chunks.

## Log
- 01:18 Regression: cond_1 and tfeat_1.3333+score_1 bit-identical after adding the new parts; all new methods run.
- 02:39 Circuit analysis (`bench/cond_circuit.py`, csv in SiT-L_2_16seeds/analysis/cond_circuit.csv). Removing one modulation
  component's class dependence on saved w=4 states (seeds 0-15, s_t 0.06/0.16/0.29); share of the shared vs seed-specific push:
  - No small circuit: the push is spread over many blocks/chunks. Gates carry the most (all:gate_msa 0.29 shared / 0.38 specific
    at block 15; all:gate_mlp 0.34 / 0.32).
  - gate_msa leans seed-specific (ratio ~0.8): it shapes individual samples.
  - shift_mlp is the clearest collapse carrier: all blocks 0.26 shared / 0.12 specific at block 15 (ratio 2.2); blocks 12-15 alone
    0.22 / 0.08 (2.6). shift_msa leans the same way (0.10 / 0.05).
  -> circuit candidates: damp only the shift chunks' class dependence (mcond with mchunks 8 = shift_mlp, 9 = shift_msa+shift_mlp).
- 03:57 Round 1 screen (seeds 16-23, vs cond curve at same precision; DINO / CLIP):
  cads_0.25 -0.13* / +0.11; cads_0.1 -0.13* / +0.02; cnoise_1,2 -0.25* / -0.23* (no diversity at all, the network ignores
  embedding noise of this size); tshuffle -0.06 / -0.13; tshuffle+score+late_6 -0.17* / -0.02; reference tfeat+score+late_6+
  smooth_0.1 -0.03 / +0.12. (* = interval excludes 0.) Nothing above the curve. CADS raises CLIP a bit, not DINO.
- 04:05 Added `wall_w` (guidance w at all steps) and `slg_s` (skip-layer guidance, blocks 10-12 bypassed, extra pass): lower class
  guidance for diversity + class-agnostic quality guidance for fidelity (autoguidance/SLG idea). Not compute-matched.
- 04:37 Round 2 (circuit candidates, seeds 16-23). Fixed the curve test to include w4 as cond_0 (it was clamping at cond_0.5, which
  made conditions with precision above cond_0.5 look bad). mcond on shift_mlp (all blocks, window, full removal) vs w4: DINO +0.09
  [+0.06,+0.11], CLIP +0.11 [+0.05,+0.19], precision retention +0.10 [0.00,+0.21]; shift_mlp+shift_msa vs w4: DINO +0.11, CLIP
  +0.17, layout +0.13, precision +0.00. i.e. more diverse than w4 with no fidelity loss. Blocks 12-15 only: ~nothing.
  Round 1 re-scored with the fixed curve: unchanged conclusions (CADS DINO -0.13*, tshuffle -0.06).
- 04:40 Reordered: extend both mcond candidates to seeds 24-47 (never used for selection) first, then round 4 (mcond follow-ups),
  then round 3 (skip-layer guidance).
- 06:00 HELD-OUT CHECK PASSED. mcond on the shift chunks, seeds 24-47 (never used for selection):
  shift_mlp (mchunks_8) vs w4: DINO +0.07 [+0.05,+0.08], CLIP +0.12 [+0.09,+0.14], precision ret. -0.02 [-0.08,+0.05];
    vs cond curve: DINO +0.05 [+0.00,+0.08], CLIP +0.08 [-0.01,+0.14].
  shift_mlp+shift_msa (mchunks_9) vs w4: DINO +0.11 [+0.09,+0.13], CLIP +0.17, layout +0.12, precision ret. -0.05 [-0.11,+0.01];
    vs cond curve: DINO +0.07 [+0.02,+0.11], CLIP +0.09 [-0.00,+0.18].
  Pooled 16-47: both significantly above the cond curve on DINO and CLIP. First semantic gain that survives held-out seeds.
  Effect is modest and sits at the high-fidelity end (precision ret. 0.95-0.98, DINO 0.07-0.11).
- 06:01 Visual check (comparisons/circuit_*.jpg, seeds 16-23): shift-chunk damping makes targeted content changes with no visible
  artifacts (macaw seed 17 blue close-up -> scarlet macaw in palms; macaws zoom out into wider scenes; retriever pose changes), while
  most samples stay close to w4: consistent with a modest effect at near-w4 fidelity.
- 07:38 Rounds 4-5 (seeds 16-23; vs cond curve DINO / CLIP; precision ret.):
  mcond all chunks EXCEPT gates (mchunks_27 = shifts+scales): +0.18* / +0.20*, prec 1.02, DINO 0.18, CLIP 0.20  <- best screen
  mcond all 6 chunks: +0.09 / +0.12, prec 0.57 (gates' class dependence carries the fidelity; matches the circuit analysis:
    gates lean seed-specific)
  mchunks_9 + smooth_0.1: +0.09 / +0.14*, prec 1.00 (smoothing changes little); mcond_2 (past null) on mchunks_9: -0.10 / -0.01;
  mchunks_9 + score_1: -0.13* / -0.20*; mchunks_27 + score_1: 0.00 / -0.09; mchunks_9 + tfeat: -0.07 / -0.06. Combining with
  score or tfeat hurts.
- 07:40 Held-out check of mchunks_27 on seeds 24-47 started; then a gates-only control (mchunks_36) on 16-23.
  Skip-layer guidance round (wall/slg) not run: no time.
- 08:22 Held-out check of mchunks_27 (shifts+scales) FAILED: seeds 24-47 precision ret. 0.79 (screen said 1.02), DINO 0.21,
  CLIP 0.27; vs cond curve DINO -0.02 [-0.14,+0.08], CLIP -0.09. On the curve, like everything else; the screen was a lucky draw.
  Only the shift-only variants (mchunks_8, mchunks_9) remain above the curve on held-out seeds.
- 08:36 Gates-only control (mchunks_36, seeds 16-23): precision ret. 0.71, DINO 0.44, CLIP 0.56; vs curve +0.05 / +0.11 (n.s.).
  Removing only the gates' class dependence behaves like condition damping: the gates carry most of the class effect overall
  (diversity and fidelity together), not fidelity specifically. Revises the 07:38 reading.
- 08:40 Stopped: no time for another properly held-out experiment.
- (evening 11/24) Skip-layer guidance screen (seeds 16-23), blocks 10-12 skipped, every step: precision ret. collapses (w2+slg1
  -0.05, w2+slg2 -0.10, w3+slg1 0.12) while DINO 0.19-0.57. Images become flat, smooth, illustration-like with saturated colors
  (comparisons/slg_seed_17.jpg): the push away from the degraded model removes real texture. Not useful at these settings.
  Plain w=3 (every step): precision ret. 1.00, DINO 0.10, CLIP 0.14 on the screen -- about the same as mcond shift damping
  (1.00 / 0.11 / 0.17). So the shift-damping gain over the cond curve may just be "lower global w beats cond damping at the
  high-fidelity end". Running w=3 on held-out seeds 24-47 to compare directly.
- (21:46) Plain w=3 on held-out seeds 24-47: precision ret. 0.95, DINO 0.08, CLIP 0.12, layout +0.24 vs w4; vs cond curve DINO
  +0.04 [-0.00,+0.08], CLIP +0.05 (n.s.). Paired vs w=3: shift_mlp damping DINO -0.02*, CLIP -0.01, precision +0.03 (n.s.), layout
  -0.24* -> no better than lowering w. shift_mlp+shift_msa damping DINO +0.03 [+0.00,+0.05], CLIP +0.04 [+0.01,+0.08], precision
  -0.00, layout -0.12* -> a small real semantic gain over w=3 at equal fidelity, traded against layout.
  REVISED CONCLUSION: most of the overnight "beats cond damping" result is "lower global w beats cond damping at the high-fidelity
  end"; the circuit method adds a small extra, shifting diversity from layout toward semantics.
- (22:06) Per-head analysis (`bench/head_circuit.py`; attribution patching on all 384 heads, checked by exact patching of the top
  16: agreement within ~20-30% at block 15, ordering mostly preserved at the output). No collapse heads: at block 15 the most
  shared-leaning heads (blocks 14-15) have shared/specific ~1.4 (e.g. 14.8: 0.039 / 0.026), weaker than the shift_mlp chunk
  (2.2-2.6); the top 16 heads hold ~half of block 15's shared push. At the output those heads carry shared ~ specific. Late heads
  (16-23) counteract the push in both parts. Screening hcond on the top 8 / 32 / 64 heads (ranking by block-15 shared - specific).
- (23:01) Per-head damping screen (seeds 16-23, hcond_1.3333 = head output fully replaced by the null branch's during the window):
  top 8 heads: precision 1.07, DINO 0.02, CLIP 0.02 (barely does anything); top 32: 1.00 / 0.07 / 0.08 (vs w=3: DINO -0.02,
  CLIP -0.07, n.s.; layout -0.30*); top 64: 0.90 / 0.08 / 0.09 (vs cond curve DINO -0.13). No better than lowering w, so no
  held-out run. Consistent with the analysis: no head is collapse-specific enough.

# Session 3 (2026-11-25, from 00:36)
- Vendi (cosine, DINO and CLIP features) added to guidance_intervene_eval.py, paired_vs_cond.py and points.py. Rerun of the
  key comparisons (paired_tests/vendi_*.txt): Vendi agrees with the pairwise-distance measures everywhere (same signs, similar
  sizes, same significance); no conclusion changes.
- CORRECTION: with w4 on the curve, score damping alone is above the condition-damping curve on fresh seeds (32-47: DINO
  +0.13 [+0.07,+0.16], Vendi DINO +0.11, precision ret. 1.00). Earlier fresh-seed "vs curve" numbers were computed before the
  w4 fix and clamped at cond_0.5. Held out 24-47 (paired_tests/score_vs_wall3_24_47.txt): score_1 precision ret. 0.98, DINO
  0.12, CLIP 0.15, layout +0.96 vs w4. vs w=3: DINO +0.04 [+0.01,+0.06], layout +0.72, precision +0.03 (n.s.). vs shift damping
  (mchunks_9): same semantics (+0.01), layout +0.84. => score damping matches or beats every later method at the high-fidelity
  end; the circuit method's small gain over w=3 is matched by it.
- Fetching the ImageNet training split for these 8 classes (embeddings only) into outputs/real_imagenet_train/.
- Joint sampling (`bench/joint_sample.py`): all seeds of a group sampled together; during s_t < 0.3, samples of a class repel
  each other (RBF kernel on predicted clean latents, median bandwidth; push = s * ||Delta_i|| along the repulsion direction).
  Check: joint plain CFG is bit-identical to the per-seed w4 samples. Screening rep_1, rep_3, rep_1+score_1, rep_3+score_1 on
  group 16-31; held-out group 32-47.
- 02:30 Training-set fetch: the connection dropped to ~0.27 MB/s (Cloudflare and Hugging Face alike; ~4-5 MB/s earlier), so
  the full 147 GB cannot finish. Shards are random slices of ImageNet, so a partial fetch is still an unbiased per-class
  sample: keep both workers running and merge whatever has arrived at ~09:30 (recording the shard count).
- 02:42 Joint latent repulsion, group 16-31 (paired_tests/joint_dev.txt): rep_1 precision 1.02, DINO 0.01, CLIP 0.05, layout
  +0.37 vs w4 (layout for free, no semantics); rep_3 0.89 / 0.07 / 0.13, DINO vs curve -0.06; rep_1+score_1 0.91 / 0.11 / 0.18
  (+0.02 DINO over score alone, n.s.); rep_3+score_1 0.76 / 0.15 / 0.22, DINO vs curve -0.12*. Repulsion between predicted
  latents diversifies layout (what latent distance mostly measures), not semantics. Next: repulsion in mid-block feature space.
- 04:07 Feature-space repulsion at block 12 (frep, token activations), group 16-31 (paired_tests/joint_frep.txt): frep_0.25
  precision 0.93, DINO -0.02, layout +0.27; frep_0.5 0.87 / 0.03 / +0.53; frep_1 0.24 / 0.25 / +1.01. All below the curve on
  DINO (-0.11 to -0.19*). Token activations are organised by position, so their distances mostly reflect layout too.
  Next: repulsion in DINOv2 space of the decoded predicted image (gradient through VAE decoder + DINOv2, not SiT).
- 04:27 DINOv2-space repulsion (drep) smoke test: ~4 GB peak, ~75 min per 16-seed condition; s=1 barely changes samples (mean
  latent change 0.014: the pulled-back DINO gradient is mostly detail SiT removes again), s=10 gives 0.12. Screening drep_10
  and drep_25 on group 16-31.
- 09:36 Connection recovered; training-set fetch complete: 1,300 images per class, embeddings only
  (outputs/real_imagenet_train/dino.npz). The DINOv2-space repulsion run (drep_10) had not finished its first 16-seed
  condition after 5 h (process swapped out): stopped, not evaluated.
- 09:58 Re-scored held-out 24-47 against the 1,300-image training reference (paired_tests/train_ref_heldout_24_47.txt):
  score_1 precision ret. 1.00, DINO +0.11 [+0.08,+0.14] and CLIP +0.14 vs the cond curve, vs w=3 precision +0.24 [+0.10,+0.37];
  w=3 0.77 (on the curve, -0.00); shift damping 0.77 (+0.03 n.s.); shifts+scales 0.58 (-0.05). With the finer reference the
  earlier edges of w=3 and shift damping disappear; score damping is the clear winner. Figure: figures/frontier_heldout.png.
  (knn_radius rewritten without the (N, N, D) broadcast: same values, needed at N = 1300.)
- 10:02 CORRECTION to 09:58: on seeds 16-31 the same training-reference comparison reverses (score_1 precision ret. 0.81,
  w=3 0.98). Pooled 16-47 (paired_tests/train_ref_pooled_16_47.txt): score_1 0.91 / DINO 0.12, vs cond curve DINO +0.08
  [+0.03,+0.13], CLIP +0.10 [+0.04,+0.17], vs w4 precision -0.09 (n.s.); w=3 0.85 / 0.09 (+0.02 n.s.); shift damping 0.82 /
  0.11 (+0.03 n.s.). Score damping is still the one method significantly above the curve, but modestly; precision varies a
  lot between 16-seed sets. Joint sampling re-scored with the training reference (paired_tests/train_ref_joint_16_31.txt):
  nothing above the curve. RESULTS.md 12.7 and the figure use the pooled numbers.
- (later, 11/25) Score-damping combinations re-scored against the training reference, seeds 16-47 pooled
  (paired_tests/train_ref_score_combos_16_47.txt; ~730 MB peak, 2 threads): tfeat+score 0.42 / DINO 0.43 (-0.04 vs curve,
  CLIP -0.13*); +late_6 0.66 / 0.25 (-0.00); +late_6+smooth 0.61 / 0.32 (-0.01); cond_1+late_6+smooth 0.43 / 0.38 (-0.08*).
  All on or below the curve; only score damping alone is above it.

# Session 4 (2026-11-26, overnight): directions 1-7 of RESULTS.md 13 / the follow-up plan
Queue: logs/overnight.sh, then logs/overnight2.sh from 05:12 (switch sweep cut to seeds 0-3 so 3 and 5 fit before 10:00).
All new GPU scripts were checked first: switch_sweep restart from saved x_t reproduces saved w4 finals exactly (0.0, k = 0 and 20);
extra_dirs cond_1+rec = SiT-L_2_confirm cond_1 seed 32 exactly; attn_capture hooks leave all 16 samples bit-identical.
- Dir 1 (paired_tests/window_fidelity_16_47.txt, train_ref_windows_*): seeds 16-47, 1,300-image reference. Precision w1 0.26,
  early-only 0.34, late-only 0.51, w4 0.73: late guidance carries most of the precision gain. DINO diversity 0.655 / 0.370 / 0.370 /
  0.244 (real images 0.494): each window alone removes 0.69 of the loss. Typicality (cos. distance to real class mean) 0.50 / 0.255 /
  0.235 / 0.175 (real 0.30): guided samples are MORE typical than real photos. Among samples inside the real manifold under both
  w1 and late-only, late guidance still cuts DINO diversity 0.62 -> 0.38: real diversity loss, not only repair. Coverage rises with
  guidance (0.008 -> 0.053; w1 samples are mostly off-manifold). (Fixed: recall/coverage intervals now from half-samples.)
- Dir 2b (SiT-L_2_16seeds/analysis/probe_identity.csv): ridge probes, top-8 within-class DINO directions of the final image. At w4
  identity is readable from pooled blocks 15-17 by s_t 0.1-0.2 (R2 0.19-0.22), at w1 only ~0.3; dino(x_hat) catches up at 0.3-0.4;
  blocks 0-7 and x_t never. R2 is relative to each run's own (smaller at w4) variance. No intervals yet.
- Dir 2a (paired_tests/switch.txt, seeds 0-3): layout half-commits at median s_t 0.09 and is 0.95 committed by 0.31; semantics
  half-commit at 0.10 but stay open: guidance from 0.31 on still moves semantics 0.66 of the way to the w4 outcome, from 0.57 on 0.35.
  Guidance from s_t 0.12 on: DINO div 0.260 (w4 0.242) with layout V 0.612 (w1 0.661), precision 0.84. Layout is lost only to
  guidance before ~0.2; semantic diversity to guidance at any time up to ~0.6. Hybrids sit far off the w1-w4 segment in DINO (0.4-0.6).
- Dir 4 (paired_tests/along_across.txt): in the window cond damping is mostly a weakening along w4's class signal (cond_0.5 along
  -0.49, across 0.21); after the window every condition is mostly redirection (cond_0.5 -0.11 / 0.31, score -0.10 / 0.35), mostly in
  the seed-specific part. Along stays negative: no compensation.
- Dir 6 (paired_tests/attn.txt, seeds 0-7): the class changes attention in blocks 8-23 (cond vs uncond L1 0.32-0.40 of 2; blocks 0-7
  0.08); top class-dependent heads all in blocks 15-17. Guidance barely changes attention (entropy -0.003..-0.015, distance <= -0.1
  patch) and does not make seeds attend to the same places in the window (cross-seed similarity 0.517 vs 0.521 mid blocks). No CIs.
- Dir 7 (paired_tests/gaussian_theory_variants.txt): Gaussian-CFG predictions in the low band depend strongly on covariance
  estimation (V_w4/V_w1 0.64-1.00 over 5 variants; one matches SiT). Robust: at w1 a linear map from noise predicts 58-75% of each
  sample's final layout; Gaussian w4 dynamics over-amplify seed deviations (per-seed R2 -0.4 to -78); SiT's Delta turns seed-specific
  faster than the Gaussian's (S at s_t 0.3: Gaussian 0.15-0.41, SiT 0.08-0.13).
- Dir 3 (paired_tests/extra_dirs_32_47.txt, points; seeds 32-47 held out from the fit): cond_1+rec reproduces cond_1 exactly. Ablating
  the extra late-block directions E after the window: DINO recovery 0.56 -> 0.53 (-0.02 [-0.06,+0.01]), no different from ablating
  control directions O (E - O: -0.01 [-0.05,+0.03]); both cost precision. Injecting E into w4: DINO +0.04 [+0.02,+0.06] at -0.17
  precision retention, on the cond curve. => The extra effective rank under condition damping is not what carries its semantic
  diversity (the 11.2 correlation is not causal, at the pooled level after the window).
- Dir 5 (paired_tests/specswap_16_47.txt, points; seeds 16-47 pooled, 1,300-image reference). Precision retention: score_1 (shared
  push removed) 0.91, spec_1 (seed-specific part removed, shared kept) 0.75 (-0.25 [-0.36,-0.13] vs w4), swap_1 (another seed's
  seed-specific part) 0.67 (-0.33 [-0.43,-0.22]; vs score_1 -0.24 [-0.37,-0.13]), scalar_1 (both) 0.54. Layout vs w4: spec_1 -0.15
  [-0.19,-0.11], i.e. LESS layout diversity than w4. => Early fidelity leans on the seed-specific part (removing it costs ~0.25,
  removing the shared part ~0.09), and it has to match the sample (a swapped one is worse than none). The shared part carries the
  layout collapse; the seed-specific part keeps samples apart. This is why score damping sits above the curve.
  CORRECTION to the 09:05 reading on seeds 16-31 alone (score_1 and spec_1 both 0.81 there, suggesting an even split): the pooled
  32 seeds reverse it. Seed-set variance in precision again.
- Not yet done: intervals for the probes (2b) and the attention cross-seed similarity (6); switch sweep only on seeds 0-3 (2a);
  Gaussian theory not identifiable from 48 samples per class (7).

# Session 5 (2026-11-26 afternoon): patching between the w1 and w4 runs of one seed (bench/patch_sweep.py, patch_analysis.py)
Full residual-stream patching is degenerate here: both runs evaluate the same network with the same t and y embeddings (guidance
only mixes outputs), so a patched residual makes every later block and the output equal the donor's, for any block. Instead:
block-write patching (block b's attention+MLP contribution from the donor, 5 steps from k), a 5-step velocity transplant (upper
bound), and swapping one frequency band of x_t at step k. Check: donor = recipient reproduces the samples (0 / 4e-5 / 5e-6).
Screen, seeds 0-1 (16 samples/cell; paired_tests/patch_screen_0_1.txt). lambda = median move toward the donor's outcome:
- x_t low band (<= 4 cycles) carries most of the guided outcome: w4 -> w1 at s_t 0.24: semantics 0.73 [0.57,0.83], layout 0.84
  (a full x_t swap there gives ~0.83, switch sweep). Mid/high bands ~0.0-0.15 at 0.24; by 0.41 high band 0.22, mid 0.15.
- Block writes: each <= 0.10 semantics; as a share of the 5-step velocity transplant, blocks 13-15 carry the most (~0.3; block 15
  0.43 for r4 at 0.24), block 4 0.02-0.16, blocks 20-23 0.06-0.24. Distributed, peaking in mid blocks.
- A 5-step velocity transplant moves semantics only 0.11-0.33: the effect is spread over time.
- Undoing (w1 -> into w4) is weaker than imposing everywhere (low band at 0.24: 0.28 vs 0.73).
- r4 low band at 0.24: DINO recovery +0.19, precision 0.69 -> 0.50 (16 samples: steps of 0.06).
=> The guided semantic outcome at s_t ~0.24 is carried mostly by the coarse layout of the state, including its seed-specific
   guided part (score damping removes only the shared part, so it restores layout but little semantics).

# Session 6 (night of 2026-11-27)
- Probe intervals (paired_tests/probe_ci.txt; 300 subsets of 12 of 16 seeds, permutation null). At w4, blocks 15-16 read the
  final identity from s_t 0.16 on (12-seed lower bounds > 0; full-data R2 0.16-0.28 vs null 97.5% 0.001-0.008); blocks 4 and 10
  never. CORRECTION: the w4 - w1 difference is not significant anywhere (e.g. step 8, block 15: +0.11 [-0.03,+0.31]), so "identity
  is readable EARLIER at w4" is not supported; only "readable in mid blocks 15-16 by s_t ~0.16, before x_hat shows it" is.
  (12-seed subsample probes are weaker, so their intervals sit below the 16-seed point values: conservative.)
- Attention seed-similarity intervals (paired_tests/attn.txt): in the window, w4 - w1 is +0.004 [-0.000,+0.009] in blocks 8-15 and
  +0.000 [-0.006,+0.007] in 16-23 (no change); blocks 0-7 +0.031 [+0.027,+0.035] in the window and +0.065 after; small rises in all
  groups after the window. Attention similarity follows the layout convergence; it does not lead it.
- Patching confirmation, seeds 0-7 (paired_tests/patch_0_7.txt; each condition on its own seeds, 64 samples per cell for the
  confirmation grid; blocks 4/17/23 only on seeds 0-1). (Fixed: patch_analysis.py first took the seeds shared by ALL conditions,
  which was only 0-1.)
  * Low band of x_t, w4 -> into w1, semantics: 0.51 / 0.72 / 0.68 at s_t 0.10 / 0.24 / 0.41 (layout 0.54 / 0.86 / 0.91).
  * Split of the low band (semantics, w4 -> w1): shared 0.37 [0.29,0.47] vs seed-specific 0.16 [0.10,0.23] at 0.10;
    0.40 [0.28,0.49] vs 0.53 [0.41,0.63] at 0.24; 0.21 [0.15,0.26] vs 0.52 [0.43,0.58] at 0.41. CROSSOVER: early the shared
    class push carries the guided outcome (semantics included); from ~0.24 on the seed-specific part carries the semantics.
    Undoing (w1 -> into w4) shows the same crossover, weaker (0.09/0.05, 0.08/0.15, 0.04/0.12).
  * Undoing with the seed-specific part costs precision (0.69 -> 0.55 at 0.24 and 0.41, DINO recovery +0.04 / +0.13); with the
    shared part precision is kept (0.75 / 0.72) and no diversity comes back. Matches direction 5.
  * Block writes: each <= 0.10 semantics. Share of the 5-step velocity transplant: undoing, block 15 largest (0.30-0.41);
    imposing, blocks 10-20 about flat (0.16-0.26). "Peaks in blocks 13-15" holds only for the undo direction.
- Queued (logs/overnight5.sh): split on seeds 8-15; then split at k = 8, 16 on seeds 0-7 if it starts by 08:10.
- Switch sweep on 16 seeds (paired_tests/switch.txt, 128 samples): layout committed 0.95 [0.95,0.96] by s_t 0.31, half at median
  0.10; semantics half at 0.09 but guidance from 0.31 on still moves them 0.65 [0.63,0.66], from 0.57 on 0.35 [0.34,0.36].
  Guidance from 0.12 on: DINO div 0.269 (w4 0.228), layout V 0.760 (w1 0.819), precision 0.71 (w4 0.68). Confirms the 4-seed run.
- Low-band split on 16 seeds (paired_tests/patch_all.txt, 128 samples): semantics w4 -> w1, shared vs seed-specific: 0.41 [0.33,0.47]
  vs 0.18 [0.15,0.21] at 0.10; 0.39 [0.35,0.46] vs 0.55 [0.48,0.62] at 0.24; 0.22 [0.18,0.26] vs 0.53 [0.47,0.56] at 0.41.
  Crossover confirmed. Undoing with the seed-specific part: precision 0.68 -> 0.58 / 0.52, DINO recovery +0.07 / +0.12; with the
  shared part precision 0.69-0.70, recovery +0.03 to +0.05.
- Split at two more times (k = 8, 16; seeds 0-6, sampling stopped at 09:20 as planned): semantics w4 -> w1 shared / seed-specific
  0.43 / 0.36 at s_t 0.16, 0.25 / 0.55 at 0.33. With the 16-seed times: handover at s_t ~0.2 (about equal at 0.16, seed-specific
  leads from 0.24), the same time layout finishes committing. All GPU work ended 09:24.

# Session 7 (2026-11-27 evening): stochastic sampler (SiT's SDE, Euler-Maruyama, D(t) = sigma_t, last step Mean, 100 steps)
`guidance_intervene.py --sampler sde`: same noise per seed across conditions (CPU global generator seeded per seed); a one-entry
cache on the model function removes SiT's second network call per step (drift and score on the same x, t; checked exact: 0.0).
Run SiT-L_2_sde, seeds 16-31, stopped after 3 conditions on the user's instruction (same pattern as ODE). vs ODE on the same seeds:
- w1 -> w4: DINO 0.648 -> 0.246 (ODE) vs 0.600 -> 0.235 (SDE): guidance keeps 0.38 / 0.39 of w1's semantic diversity. Layout
  0.813 -> 0.611 vs 0.812 -> 0.646 (SDE keeps slightly more). Precision 0.24 -> 0.73 vs 0.20 -> 0.66.
- score_1: layout fully recovered under both (0.816 / 0.815), DINO recovery 0.10 / 0.08, precision retention 0.81 / 0.98 (the ODE
  value on these seeds is the known low draw; 0.91 pooled over 16-47).
=> Stochastic sampling does not change the picture: semantic diversity is lost to the same extent, and score damping does the same
   thing. Condition-damping curve under SDE not run (cond_0.5 stopped mid-run; partial seeds in SiT-L_2_sde/cond_*).
- VAE check (bench/vae_check.py, paired_tests/vae_check.txt). (a) 400 real validation images through VAE encode + decode: DINO
  diversity 0.473 -> 0.490, CLIP 0.195 -> 0.193, reconstruction DINO cosine 0.957: the VAE does not compress semantic diversity.
  (b) Latent RMS: real 0.807, w1 0.764, w4 0.892. Rescaling w4 latents to w1's RMS: DINO 0.244 -> 0.242; w1 scaled up to w4's:
  0.655 -> 0.656. Decoded diversity does not depend on the latent scale. => the diversity loss is in SiT's latents, not the decoder.
- Realistic diversity (bench/realistic_diversity.py, paired_tests/realistic_diversity_16_47.txt; seeds 16-47; per class 32 real
  images held out as the "real sample", all scored against the other 1,268; DINO, k = 3). Real sample: precision 0.76, coverage
  0.070, realistic Vendi (Vendi over on-manifold samples, per 32) 7.36, KID ~0. w4: 0.73 / 0.054 / 3.01 / KID 6974 (worst of all
  conditions, worse than w1's 6375). w1: 0.26 / 0.007 / 5.40. Best KID: cond_1 2805, late-only (scalar_1) 2821, cond_0.75 3043.
  score_1: 0.68 / 0.054 / 3.64 (+0.63 [+0.34,+0.91] realistic Vendi over w4 at equal coverage).
  => Counting only realistic images, w4 keeps 0.56 of w1's diversity (raw DINO distance: 0.37): about a third of the measured
     loss was off-manifold spread. But w4 has only 0.41 of real images' realistic diversity at near-real precision, and is the
     poorest match to the real distribution of everything tested; moderate damping matches it best.
- Realistic frontier (bench/realistic_frontier.py, paired_tests/frontier_realistic_16_47.txt, figures/frontier_realistic.png):
  realistic Vendi vs the condition-damping curve at equal precision, seeds 16-47, paired half-sample intervals (~ +-0.7).
  Score damping +0.12 [-0.71,+0.71]: NOT above the curve on realistic diversity (its earlier edge was layout and raw DINO spread).
  tfeat +0.79 [+0.12,+1.44] (one of 17 conditions tested; tfeat failed held-out replication before, so likely chance). Clearly
  below: shared push only -0.89, swapped seed part -1.48, guidance early only -1.51. The metric is noisy: cond_0.5 (0.56, 3.69)
  and cond_0.75 (0.57, 4.62) differ by ~1 at equal precision. Every method sits far from the real images (precision 0.76,
  realistic Vendi 7.36): the frontier spans 3.0 (w4) to ~4.7 (precision 0.43-0.57).

# Session 8 (night of 2026-11-28)
- Switch sweep scored with realistic measures (paired_tests/realistic_switch_0_15.txt; seeds 0-15, so indicative): guidance on from
  s_t 0.12 (w1 before): precision 0.71 (w4 0.68), realistic Vendi/32 5.92 (w4 4.83), KID 5385 (w4 8216); from 0.31: 0.62 / 6.97 /
  2841. Guidance in the first ~12% of sampling buys no realism and only costs realistic diversity (the shared-push phase).
- When the subcluster is chosen (bench/subcluster_commit.py, paired_tests/subcluster_commit.txt; switch sweep seeds 0-15, K = 8):
  69% of samples change subcluster between w1 and w4. Of those, guidance only before s_t = 0.06 / 0.12 / 0.31 / 0.57 already puts
  0.24 / 0.45 / 0.65 / 0.88 in the w4 subcluster; guidance only after 0.12 / 0.31 / 0.57 still puts 0.45 / 0.34 / 0.22 there.
  Subcluster choice is spread over most of sampling (half by ~0.12), unlike layout (fixed by ~0.2). Implication for steering: a
  window ending at 0.3 can be partly undone by later guidance.
- Subcluster steering, attempt 1 (bench/subcluster_steer.py; DINO-centroid gradient through VAE + DINOv2, low-passed, s_t 0.06-0.3;
  seeds 16-19; paired_tests/steer_screen_16_19.txt): no effect. Hit rate 0.25 at s = 1/2/4 vs 0.28 for w4 with the same targets;
  cosine to target +0.01. Latents change ~10% but not toward the target (the decoded x_hat early on is too blurry for DINO's
  gradient to carry semantics). Memory: 12 GB with 4-image backward chunks; 5.6 GB with one image at a time. Stopped.
- Attempt 2 (bench/ref_steer.py; ILVR-style: at step k move the low band of x_t toward a random validation image's, t z_ref +
  (1-t) eps; paired_tests/ref_screen_16_19.txt, seeds 16-19): alpha 1 at s_t 0.24 raises the hit rate 0.12 -> 0.38 (+0.25
  [+0.19,+0.34]) with little copying (+0.03 cosine to own reference), but precision 0.66 -> 0.53, realistic Vendi 1.60 -> 1.41,
  subclusters used 2.66 -> 2.87. alpha 0.5: ~no effect; at s_t 0.12 precision falls to 0.41. Steers, but buys no realistic
  diversity. Stopped at the checkpoint.
- Guidance-start family (gstart_t: w = 1 before t, w = 4 after; new in guidance_intervene.py, gstart_0.3 == scalar_1 exactly).
  Held-out seeds 16-31 vs w4 (paired_tests/realistic_diversity_16_31.txt): realistic Vendi/32 +0.68 [+0.25,+1.12] (from 0.06),
  +0.98 [+0.54,+1.45] (0.12), +1.07 [+0.51,+1.66] (0.20); KID -843 / -1627 / -2742; precision 0.71 / 0.69 / 0.66 (w4 0.73);
  coverage -0.003. Replicates the switch-sweep result (seeds 0-15) with a small precision cost here. w = 5 from 0.12: precision
  0.67, no better. vs the condition-damping curve all +0.14 to +0.45, n.s. (realistic Vendi too noisy at 16 seeds, +-1.2).
- Subcluster occupancy (bench/subcluster_occupancy.py, half-sample intervals; paired_tests/subcluster_occupancy_16_31.txt), vs w4:
  effective subclusters among on-manifold samples +0.70 [+0.46,+0.93] (guidance from 0.06), +0.52 [+0.22,+0.86] (0.12), +0.59
  [+0.20,+0.98] (0.20), +0.98 (from 0.3), score_1 +0.38 [+0.05,+0.72]; overall (all samples) gstart_0.12 +0.09 n.s. The extra
  realistic diversity from skipping early guidance is realistic samples landing in more of the real subclusters: the early shared
  push herds them into the dominant one. (Fixed: the first version used a bootstrap, biased for these set-based counts.)
- Queued: gstart 32-47 (pooled analysis to follow), then seeds 48-63 for w4, cond curve, score_1, gstart_0.12/0.2 (logs/seeds48.sh).
- Guidance start, POOLED seeds 16-47 (paired_tests/realistic_diversity_16_47.txt, subcluster_occupancy_16_47.txt,
  frontier_realistic_16_47.txt): vs w4, realistic Vendi/32 +0.27 [+0.04,+0.51] (from 0.06, precision 0.73 = w4), +0.66 [+0.41,
  +0.92] (0.12, precision 0.66), +0.92 [+0.61,+1.25] (0.20, precision 0.65); on-manifold subclusters +0.47 / +0.63 / +0.86.
  vs the condition-damping curve: +0.23 / -0.04 / +0.14, all n.s. CORRECTION to the 16-seed readings ("guidance before 0.12 is
  free"): pooled, only the first ~6% is free; later starts cost precision at the curve's rate. Every method tested lies on one
  realistic frontier within noise. Seeds 48-63 running for the frontier conditions.
- Robustness (RD_K=5 in realistic_diversity.py; paired_tests/realistic_diversity_16_47_k5.txt): with a 5-NN manifold radius, nothing
  is above the condition-damping curve either (score_1 +0.23, gstart_0.12 +0.37, all n.s.); tfeat's +0.79 at k = 3 becomes -0.33
  (noise). Below the curve at both k: swapped seed part, guidance early only, cond_1 + late 6.
- Realistic frontier on 46 seeds (16-61; seeds 48-61 added for w4, cond curve, score_1, gstart_0.12/0.2; sampling stopped at 09:45
  as planned; paired_tests/frontier_realistic_16_61.txt, figures/frontier_realistic_16-61.png). Intervals now ~+-0.4. vs the
  condition-damping curve: score_1 +0.23 [-0.19,+0.61], gstart_0.2 +0.19 [-0.23,+0.58], gstart_0.12 +0.14 [-0.27,+0.51]; real
  sample +3.71. The curve is now monotone. All methods on one realistic frontier, confirmed. (realistic_diversity.py: the held-out
  real sample now has max(32, n_seeds) images; absolute realistic Vendi is lower at 46 seeds than at 32, compare within a table.)
