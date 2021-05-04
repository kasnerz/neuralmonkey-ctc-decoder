#!/bin/bash

OUT_DIR="out/pipeline/kd"

# CHECKPOINT_TYPES=("avg" "best")
CHECKPOINT_TYPES=("avg")
# DATASET_TYPES=("val" "test")
DATASET_TYPES=("test")
BEAM_SIZES=(1 5 10 20)
# BEAM_SIZES=(1)

LANG1=$1
LANG2=$2
EXTRA_PREFIX="kd"


for CHECKPOINT_TYPE in "${CHECKPOINT_TYPES[@]}"; do
    for DATASET_TYPE in "${DATASET_TYPES[@]}"; do
        for BEAM_SIZE in "${BEAM_SIZES[@]}"; do

            echo "Submitting ${LANG1}-${LANG2} beam ${BEAM_SIZE}"

            JOB_NAME="$LANG1$LANG2$EXTRA_PREFIX-$CHECKPOINT_TYPE-$DATASET_TYPE-b$BEAM_SIZE-ld$LM_SCORE_DEFAULT-nd$NULL_TRAILING_DEFAULT-rd$NULL_TOKEN_RATIO_DEFAULT"
            OUT_FILE="$OUT_DIR/$JOB_NAME"

            qsub -q "cpu-troja.q" -l 'mem_free=32G' -cwd -b y -j y -pty yes -m n -o "$OUT_FILE" -e "$OUT_FILE" -N "$JOB_NAME" ./pipeline.sh -s "$LANG1" -t "$LANG2" -c "$CHECKPOINT_TYPE" -d "$DATASET_TYPE" -b "$BEAM_SIZE" -l 0.0 -n 0.0 -r 0.0
#            qsub -q "cpu-ms.q" -l 'mem_free=32G,hostname=orion*' -cwd -b y -j y -pty yes -m n -o "$OUT_FILE" -e "$OUT_FILE" -N "$JOB_NAME" ./pipeline.sh -s "$LANG1" -t "$LANG2" -c "$CHECKPOINT_TYPE" -d "$DATASET_TYPE" -b "$BEAM_SIZE" -l 0.0 -n 0.0 -r 0.0

            # qsub -q "gpu-*.q" -l 'mem_free=32G,hostname=*dll*' -cwd -b y -j y -pty yes -m n -o "$OUT_FILE" -e "$OUT_FILE" -N "$JOB_NAME" ./pipeline.sh -s "$LANG1" -t "$LANG2" -c "$CHECKPOINT_TYPE" -d "$DATASET_TYPE" -b "$BEAM_SIZE"
        done
    done
done

