#!/bin/zsh
# Night of 2026-11-27: after the patching chain, analyse it (CPU), then extend the switch sweep to seeds 4-15 (GPU), embed,
# and rerun the switch analysis on 16 seeds. One GPU job at a time; a GPU job still running at HARD is stopped (batches are
# written atomically and skipped on rerun).
setopt NO_NOMATCH
cd "$(dirname "$0")/../../.."  # project root (this file is in outputs/guidance_delta/logs)
O=outputs/guidance_delta; L=$O/logs; P=$O/paired_tests; PY=.venv/bin/python
W=pretrained_models/SiT-L-2-256_BiliSakura_2f02393/transformer.safetensors
HARD=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-27 09:30" +%s)
log() { echo "$(date +%H:%M:%S) $*" >> $L/overnight4.log }
gpu() {
  local name=$1; shift
  [ $(date +%s) -lt $HARD ] || { log "SKIP $name"; return 1; }
  log "START $name"; "$@" >> $L/$name.log 2>&1 & local pid=$!
  while kill -0 $pid 2>/dev/null; do [ $(date +%s) -ge $HARD ] && { log "STOP $name at 09:30"; kill $pid; }; sleep 20; done
  wait $pid; local rc=$?; log "END $name rc=$rc"; return $rc
}
log "queue 4 started; waiting for the patching chain"
while pgrep -f "SiT-L_2_patch" >/dev/null; do sleep 60; done
log "patching chain finished"
( REAL=outputs/real_imagenet_train $PY bench/patch_analysis.py $O/SiT-L_2_patch > $P/patch_0_7.txt 2>&1; log "patch analysis done" ) &
gpu switch_more $PY bench/switch_sweep.py --run $O/SiT-L_2_16seeds --weights $W --out $O/SiT-L_2_switch --seeds $(seq 4 15) --ks 3 6 10 15 20 28
gpu switch_embed3 $PY bench/guidance_intervene_eval.py --run $O/SiT-L_2_switch --embed
REAL=outputs/real_imagenet_train $PY bench/switch_analysis.py $O/SiT-L_2_switch --figure > $P/switch.txt 2>&1; log "switch analysis done"
wait
log "queue 4 finished"
