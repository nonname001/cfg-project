#!/bin/zsh
# Joint-sampling screen: sample conditions for one seed group into SiT-L_2_joint, embed, and run the paired test on that group.
# usage: bench/joint_screen.sh <seed_lo> <seed_hi> <tag> cond ...
set -e
cd "$(dirname "$0")/.."
O=outputs/guidance_delta
LO=$1; HI=$2; TAG=$3; shift 3
.venv/bin/python bench/joint_sample.py --tables $O/SiT-L_2_16seeds/analysis/intervention_tables.npz \
  --weights pretrained_models/SiT-L-2-256_BiliSakura_2f02393/transformer.safetensors --out $O/SiT-L_2_joint --seeds $(seq $LO $HI) --conditions "$@"
.venv/bin/python bench/guidance_intervene_eval.py --run $O/SiT-L_2_joint --embed > /dev/null
PAIRS=(); for c in "$@"; do PAIRS+=("${c}:w4" "${c}:score_1"); done
.venv/bin/python bench/paired_vs_cond.py $O/SiT-L_2_joint $LO $HI "$@" score_1 wall_3 -- $PAIRS > $O/paired_tests/joint_$TAG.txt
.venv/bin/python bench/points.py $O/SiT-L_2_joint $LO $HI w4 score_1 wall_3 cond_0.5 cond_0.75 "$@" >> $O/paired_tests/joint_$TAG.txt
echo JOINT $TAG DONE
