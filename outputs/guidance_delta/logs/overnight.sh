#!/bin/zsh
# Overnight queue, 2026-11-26: directions 1-7 (see conversation / RESULTS.md 13). One GPU job at a time; CPU analyses run
# alongside in the background. No GPU job starts after SOFT; a running one is stopped at HARD (every script writes each
# batch atomically and skips finished batches, so a stopped job leaves only complete files and can be resumed).
setopt NO_NOMATCH
cd "$(dirname "$0")/../../.."  # project root (this file is in outputs/guidance_delta/logs)
O=outputs/guidance_delta; L=$O/logs; P=$O/paired_tests; PY=.venv/bin/python
W=pretrained_models/SiT-L-2-256_BiliSakura_2f02393/transformer.safetensors
T=$O/SiT-L_2_16seeds/analysis/intervention_tables.npz; TT=$O/SiT-L_2_tokens/analysis/token_tables.npz
TRAIN=outputs/real_imagenet_train
SOFT=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-26 08:50" +%s)
HARD=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-26 09:25" +%s)
log() { echo "$(date +%H:%M:%S) $*" >> $L/overnight.log }
early() { [ $(date +%s) -lt $SOFT ] }
memfree() { memory_pressure | tail -1 | grep -oE '[0-9]+' }

# gpu <name> <command...>: run a GPU job in the foreground of this queue, stopping it at HARD. Waits for free memory first.
gpu() {
  local name=$1; shift
  if ! early; then log "SKIP $name (past soft deadline)"; return 1; fi
  while [ $(memfree) -lt 15 ]; do log "waiting for memory ($(memfree)% free) before $name"; sleep 60; done
  log "START $name"
  "$@" >> $L/$name.log 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ $(date +%s) -ge $HARD ]; then log "STOP $name at hard deadline"; kill $pid; sleep 5; kill -9 $pid 2>/dev/null; fi
    sleep 20
  done
  wait $pid; local rc=$?
  log "END $name rc=$rc"
  return $rc
}

log "queue started; waiting for the direction-1 sampling/embedding job"
while pgrep -f "caffeinate -i zsh -c .venv/bin/python bench/guidance_intervene.py" >/dev/null; do sleep 60; done
log "direction-1 job finished"

# --- direction 1 analysis (CPU, background)
( REAL=$TRAIN $PY bench/window_fidelity.py $O/SiT-L_2_night 16 47 > $P/window_fidelity_16_47.txt 2>&1
  REAL=$TRAIN $PY bench/paired_vs_cond.py $O/SiT-L_2_night 16 47 late_1 scalar_1 score_1 wall_3 -- scalar_1:w4 late_1:w4 > $P/train_ref_windows_16_47.txt 2>&1
  REAL=$TRAIN $PY bench/points.py $O/SiT-L_2_night 16 47 w4 late_1 scalar_1 score_1 wall_3 cond_0.5 > $P/train_ref_windows_points_16_47.txt 2>&1
  echo "$(date +%H:%M:%S) dir1 analysis done" >> $L/overnight.log ) &

# --- checks (each gates its experiment)
gpu check_switch $PY bench/switch_sweep.py --run $O/SiT-L_2_16seeds --weights $W --out $O/SiT-L_2_switch --seeds 0 --ks 15 --check; SW_OK=$?
gpu check_extra $PY bench/extra_dirs.py --sample --tables $O/SiT-L_2_extra/extra_tables.npz --itables $T --weights $W --out $O/SiT-L_2_extra --seeds 32 --conditions cond_1+rec
$PY -c "
import numpy as np, sys
a = np.load('$O/SiT-L_2_extra/cond_1+rec/seed_32.npz')['final']; b = np.load('$O/SiT-L_2_confirm/cond_1/seed_32.npz')['final']
d = np.abs(a - b).max(); print('extra rec vs confirm cond_1, seed 32: max diff', d); sys.exit(0 if d < 1e-4 else 1)" >> $L/check_extra.log 2>&1; EX_OK=$?
gpu check_specswap $PY bench/guidance_intervene.py --tables $T --token-tables $TT --weights $W --out $O/SiT-L_2_night --seeds 16 --conditions spec_1 swap_1; SS_OK=$?
log "checks: switch=$SW_OK extra=$EX_OK specswap=$SS_OK (0 = ok)"

# --- direction 4: recompute the plain network on intervened states, then the along/across split (CPU, background)
gpu along_recompute $PY bench/along_across.py --run $O/SiT-L_2_mech --recompute --weights $W
[ $? -eq 0 ] && ( $PY bench/along_across.py --run $O/SiT-L_2_mech > $P/along_across.txt 2>&1; echo "$(date +%H:%M:%S) dir4 analysis done" >> $L/overnight.log ) &

# --- direction 6: attention, seeds 0-7 (the capture asserts the samples are unchanged)
gpu attn $PY bench/attn_capture.py --run $O/SiT-L_2_16seeds --weights $W --out $O/SiT-L_2_attn --seeds 0 1 2 3 4 5 6 7
[ $? -eq 0 ] && ( $PY bench/attn_analysis.py $O/SiT-L_2_attn > $P/attn.txt 2>&1; echo "$(date +%H:%M:%S) dir6 analysis done" >> $L/overnight.log ) &

# --- direction 7: unconditional samples, then the Gaussian theory (CPU, background)
gpu uncond $PY bench/gaussian_theory.py --sample-uncond --weights $W --out $O/SiT-L_2_uncond --n-batches 24
[ $? -eq 0 ] && ( $PY bench/gaussian_theory.py --theory --uncond $O/SiT-L_2_uncond > $P/gaussian_theory.txt 2>&1; echo "$(date +%H:%M:%S) dir7 analysis done" >> $L/overnight.log ) &

# --- direction 2a: switch sweep, seeds 0-7
KS=(3 6 10 15 20 28)
if [ $SW_OK -eq 0 ]; then
  gpu switch $PY bench/switch_sweep.py --run $O/SiT-L_2_16seeds --weights $W --out $O/SiT-L_2_switch --seeds 0 1 2 3 4 5 6 7 --ks $KS
  gpu switch_embed $PY bench/guidance_intervene_eval.py --run $O/SiT-L_2_switch --embed
  ( REAL=$TRAIN $PY bench/switch_analysis.py $O/SiT-L_2_switch --figure > $P/switch.txt 2>&1; echo "$(date +%H:%M:%S) dir2a analysis done" >> $L/overnight.log ) &
fi

# --- direction 3: extra directions, seeds 32-47
if [ $EX_OK -eq 0 ]; then
  gpu extra $PY bench/extra_dirs.py --sample --tables $O/SiT-L_2_extra/extra_tables.npz --itables $T --weights $W --out $O/SiT-L_2_extra --seeds $(seq 32 47)
  gpu extra_embed $PY bench/guidance_intervene_eval.py --run $O/SiT-L_2_extra --embed
  ( REAL=$TRAIN $PY bench/paired_vs_cond.py $O/SiT-L_2_extra 32 47 cond_1+rec cond_1+xablate cond_1+oablate w4+xinject -- \
      cond_1+xablate:cond_1+rec cond_1+oablate:cond_1+rec cond_1+xablate:cond_1+oablate w4+xinject:w4 > $P/extra_dirs_32_47.txt 2>&1
    REAL=$TRAIN $PY bench/points.py $O/SiT-L_2_extra 32 47 w4 cond_1 cond_1+rec cond_1+xablate cond_1+oablate w4+xinject > $P/extra_dirs_points_32_47.txt 2>&1
    echo "$(date +%H:%M:%S) dir3 analysis done" >> $L/overnight.log ) &
fi

# --- direction 5: shared-only and swapped seed-specific guidance, seeds 16-31, then 32-47 if time allows
if [ $SS_OK -eq 0 ]; then
  gpu specswap $PY bench/guidance_intervene.py --tables $T --token-tables $TT --weights $W --out $O/SiT-L_2_night --seeds $(seq 16 31) --conditions spec_1 swap_1
  gpu specswap_more $PY bench/guidance_intervene.py --tables $T --token-tables $TT --weights $W --out $O/SiT-L_2_night --seeds $(seq 32 47) --conditions spec_1 swap_1
  gpu specswap_embed $PY bench/guidance_intervene_eval.py --run $O/SiT-L_2_night --embed
  HI=$(ls $O/SiT-L_2_night/embeddings/swap_1 | sed 's/seed_//;s/.npz//' | sort -n | awk 'NR==1{lo=$1} {if ($1==lo+NR-1) hi=$1} END{print hi}')
  ( REAL=$TRAIN $PY bench/paired_vs_cond.py $O/SiT-L_2_night 16 $HI spec_1 swap_1 score_1 scalar_1 -- spec_1:w4 swap_1:w4 spec_1:scalar_1 swap_1:score_1 > $P/specswap_16_$HI.txt 2>&1
    REAL=$TRAIN $PY bench/points.py $O/SiT-L_2_night 16 $HI w4 spec_1 swap_1 score_1 scalar_1 > $P/specswap_points_16_$HI.txt 2>&1
    echo "$(date +%H:%M:%S) dir5 analysis done (seeds 16-$HI)" >> $L/overnight.log ) &
fi

# --- if time remains: switch sweep on seeds 8-15
if [ $SW_OK -eq 0 ] && early; then
  gpu switch_more $PY bench/switch_sweep.py --run $O/SiT-L_2_16seeds --weights $W --out $O/SiT-L_2_switch --seeds 8 9 10 11 12 13 14 15 --ks $KS
  gpu switch_embed2 $PY bench/guidance_intervene_eval.py --run $O/SiT-L_2_switch --embed
  REAL=$TRAIN $PY bench/switch_analysis.py $O/SiT-L_2_switch --figure > $P/switch.txt 2>&1
fi
wait
log "queue finished"
