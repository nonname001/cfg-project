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
