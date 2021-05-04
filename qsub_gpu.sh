#!/bin/bash
GPU_CNT=1
GPU_RAM="11G"
MEM_FREE="8G"
HOSTNAME='dll4'

VOCAB_SIZE=50000
OUT_DIR="out/knowledge_distillation"
JOB_PREFIX="kd"

LANG1="en"
LANG2="de"

# EXP_ORIG_INI="experiments/san_kombo_ctc.ini"
EXP_ORIG_INI="experiments/knowledge_distillation/san_kombo_ctc.ini"
# TMP="/net/cluster/TMP/kasner"
TMP="/lnet/express/work/people/kasner/tmp"

MODEL=""            # "-complex", ""
SHARED="-shared"    # "-shared", ""
POS="-pos"          # "-pos", ""
BPU="10"            # "10", "20"

JOB_NAME="$MODEL$SHARED$POS-b$BPU-v$VOCAB_SIZE"

EXP_INI="$TMP/$LANG1$LANG2-san_ctc-$JOB_PREFIX$JOB_NAME.ini"
cp $EXP_ORIG_INI $EXP_INI

# update variables in the INI file
sed -i -e 's/^suffix=.*$/suffix='"\"-$JOB_PREFIX$JOB_NAME\""'/g' $EXP_INI

# language
sed -i -e 's/^name="E.*$/name='"\"${LANG1^^} -> ${LANG2^^}, SAN > split states > CTC (with knowledge distillation)\""'/' $EXP_INI
sed -i -e 's/^src=.*$/src='"\"$LANG1\""'/g' $EXP_INI
sed -i -e 's/^tgt=.*$/tgt='"\"$LANG2\""'/g' $EXP_INI

# --- MODEL ---
# simple model
if [ -z "$MODEL" ]; then
    sed -i -e 's/^ff_hidden_size=.*$/ff_hidden_size=2048/g' $EXP_INI
    sed -i -e 's/^n_heads=.*$/n_heads=8/g' $EXP_INI
# complex model
else
    sed -i -e 's/^ff_hidden_size=.*$/ff_hidden_size=4096/g' $EXP_INI
    sed -i -e 's/^n_heads=.*$/n_heads=16/g' $EXP_INI
fi

# --- SHARED ---
# non-shared
if [ -z "$SHARED" ]; then
    sed -i -e '/^\[decoder\]$/,+10s/^input_sequence=<input_sequence>$/#input_sequence=<input_sequence>/' $EXP_INI
# shared
else
    sed -i -e '/^\[decoder\]$/,+10s/^#input_sequence=<input_sequence>$/input_sequence=<input_sequence>/' $EXP_INI
fi

# --- POS ---
# without positional encoding
if [ -z "$POS" ]; then
    sed -i -e 's/^use_pos=.*$/use_pos=False/g' $EXP_INI
# with positional encoding
else
    sed -i -e 's/^use_pos=.*$/use_pos=True/g' $EXP_INI
fi

# --- BPU ---
# BPU=10
if [ "$BPU" -eq 10 ]; then
    sed -i -e 's/^trainer_batches_per_update=.*$/trainer_batches_per_update=20/g' $EXP_INI
    sed -i -e 's/^batch_size=.*$/batch_size=400/g' $EXP_INI
# BPU=20
else
    sed -i -e 's/^trainer_batches_per_update=.*$/trainer_batches_per_update=40/g' $EXP_INI
    sed -i -e 's/^batch_size=.*$/batch_size=400/g' $EXP_INI
fi

cat $EXP_INI
echo $EXP_INI

set -x
qsub -l "gpu=$GPU_CNT,gpu_ram=$GPU_RAM,hostname=$HOSTNAME,mem_free=$MEM_FREE" -cwd -j y -q gpu-ms.q -m n -V -o "$OUT_DIR/$LANG1$LANG2-$JOB_PREFIX$JOB_NAME" -e "$OUT_DIR/$LANG1$LANG2-$JOB_PREFIX$JOB_NAME" -N "$LANG1$LANG2-$JOB_PREFIX${JOB_NAME//-complex/complex}" ./run_training.sh "$EXP_INI"
set +x

