#!/bin/zsh
# Replacement for the tail of overnight.sh (05:20): the switch sweep stops after seeds 0-3 so that directions 3 and 5 both fit
# before 10:00. Waits for the running switch_sweep.py, then: switch embed + analysis, extra directions (seeds 32-47), spec/swap
# (seeds 16-31). One GPU job at a time; no GPU job starts after SOFT, a running one is stopped at HARD.
setopt NO_NOMATCH
cd "$(dirname "$0")/../../.."  # project root (this file is in outputs/guidance_delta/logs)
O=outputs/guidance_delta; L=$O/logs; P=$O/paired_tests; PY=.venv/bin/python
W=pretrained_models/SiT-L-2-256_BiliSakura_2f02393/transformer.safetensors
T=$O/SiT-L_2_16seeds/analysis/intervention_tables.npz; TT=$O/SiT-L_2_tokens/analysis/token_tables.npz
TRAIN=outputs/real_imagenet_train
SOFT=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-26 09:35" +%s)
HARD=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-26 09:55" +%s)
log() { echo "$(date +%H:%M:%S) $*" >> $L/overnight.log }
early() { [ $(date +%s) -lt $SOFT ] }
memfree() { memory_pressure | tail -1 | grep -oE '[0-9]+' }
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

log "queue 2 started: switch sweep limited to seeds 0-3; deadlines 09:35 / 09:55"
# Stop the running switch sweep once seed 3 is complete (w1to4_k28 is the last file written for each seed).
while pgrep -f "bench/switch_sweep.py" >/dev/null; do
  if [ -f $O/SiT-L_2_switch/w1to4_k28/seed_3.npz ]; then log "switch: seeds 0-3 done, stopping"; pkill -f "bench/switch_sweep.py"; fi
  sleep 20
done
log "END switch"

gpu switch_embed $PY bench/guidance_intervene_eval.py --run $O/SiT-L_2_switch --embed
( REAL=$TRAIN $PY bench/switch_analysis.py $O/SiT-L_2_switch --figure > $P/switch.txt 2>&1; echo "$(date +%H:%M:%S) dir2a analysis done" >> $L/overnight.log ) &

gpu extra $PY bench/extra_dirs.py --sample --tables $O/SiT-L_2_extra/extra_tables.npz --itables $T --weights $W --out $O/SiT-L_2_extra --seeds $(seq 32 47)
gpu extra_embed $PY bench/guidance_intervene_eval.py --run $O/SiT-L_2_extra --embed
HX=$(ls $O/SiT-L_2_extra/embeddings/w4+xinject 2>/dev/null | sed 's/seed_//;s/.npz//' | sort -n | awk 'NR==1{lo=$1} {if ($1==lo+NR-1) hi=$1} END{print hi}')
( REAL=$TRAIN $PY bench/paired_vs_cond.py $O/SiT-L_2_extra 32 $HX cond_1+rec cond_1+xablate cond_1+oablate w4+xinject -- \
    cond_1+xablate:cond_1+rec cond_1+oablate:cond_1+rec cond_1+xablate:cond_1+oablate w4+xinject:w4 > $P/extra_dirs_32_$HX.txt 2>&1
  REAL=$TRAIN $PY bench/points.py $O/SiT-L_2_extra 32 $HX w4 cond_1 cond_1+rec cond_1+xablate cond_1+oablate w4+xinject > $P/extra_dirs_points_32_$HX.txt 2>&1
  echo "$(date +%H:%M:%S) dir3 analysis done (seeds 32-$HX)" >> $L/overnight.log ) &

gpu specswap $PY bench/guidance_intervene.py --tables $T --token-tables $TT --weights $W --out $O/SiT-L_2_night --seeds $(seq 16 31) --conditions spec_1 swap_1
gpu specswap_embed $PY bench/guidance_intervene_eval.py --run $O/SiT-L_2_night --embed
HI=$(ls $O/SiT-L_2_night/embeddings/swap_1 2>/dev/null | sed 's/seed_//;s/.npz//' | sort -n | awk 'NR==1{lo=$1} {if ($1==lo+NR-1) hi=$1} END{print hi}')
( REAL=$TRAIN $PY bench/paired_vs_cond.py $O/SiT-L_2_night 16 $HI spec_1 swap_1 score_1 scalar_1 -- spec_1:w4 swap_1:w4 spec_1:scalar_1 swap_1:score_1 > $P/specswap_16_$HI.txt 2>&1
  REAL=$TRAIN $PY bench/points.py $O/SiT-L_2_night 16 $HI w4 spec_1 swap_1 score_1 scalar_1 > $P/specswap_points_16_$HI.txt 2>&1
  echo "$(date +%H:%M:%S) dir5 analysis done (seeds 16-$HI)" >> $L/overnight.log ) &
wait
log "queue 2 finished"
