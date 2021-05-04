#!/bin/bash

source /lnet/spec/work/people/kasner/virtualenv/tensorflow-1.10-cpu/bin/activate

if [ "$#" -lt 8 ]; then
    echo "Usage: ./generate_output LANG1 LANG2 MODEL_TYPE DATA_TYPE CHECKPOINT_TYPE LM_WEIGHT NULL_TRAIL_WEIGHT NT_RATIO_WEIGHT [OUT_FILE]"
    echo "DATA_TYPE: val, valA, valB, test"
    echo "CHECKPOINT_TYPE: best, avg"
    exit;
fi

LANG1=$1
LANG2=$2
MODEL_TYPE=$3
DATA_TYPE=$4
CHECKPOINT_TYPE=$5
VOCAB_SIZE=50000
LM_NGRAMS=5
BEAM_SIZE=10
DECODE_LAYER_INDEX=6

EXP_FILENAME="$LANG1$LANG2-san_ctc-kombo-$MODEL_TYPE-v$VOCAB_SIZE"
EXP_PATH="experiments/$EXP_FILENAME"
EXP_ORIG_INI="$EXP_PATH/experiment.ini"
EXP_ORIG_DATA_INI="experiments/san_kombo_ctc_data.ini"

LM_MODEL="../lm/news.$LANG2.$LANG1$LANG2.$VOCAB_SIZE.$LM_NGRAMS.bin"
# LM_MODEL="../lm/wmt_data.$LANG2.$VOCAB_SIZE.$LM_NGRAMS.bin"
LM_WEIGHT=$6
NULL_TRAIL_WEIGHT=$7
NT_RATIO_WEIGHT=$8

SUFFIX="wl$LM_WEIGHT-wt$NULL_TRAIL_WEIGHT-wr$NT_RATIO_WEIGHT"

if [ -z "$9" ]; then
    OUT_FILE="results/val/$EXP_FILENAME-$SUFFIX"
else
    OUT_FILE="$9"
fi

SPM_MODEL="/lnet/spec/work/people/kasner/spm/models/sp.$LANG1$LANG2.$VOCAB_SIZE.model"

TMP="/net/cluster/TMP/kasner"
EXP_INI="$TMP/$EXP_FILENAME-$SUFFIX.ini"
EXP_DATA_INI="$TMP/$EXP_FILENAME-$SUFFIX-data.ini"

# do not modify original config files
cp $EXP_ORIG_INI $EXP_INI
cp $EXP_ORIG_DATA_INI $EXP_DATA_INI

# update variables in the INI file
sed -i -e 's/^src=.*$/src="'"$LANG1"'"/' $EXP_DATA_INI
sed -i -e 's/^tgt=.*$/tgt="'"$LANG2"'"/' $EXP_DATA_INI

if [ "$DATA_TYPE" == "val" ]; then
    sed -i -e 's/^src_test_name=.*$/src_test_name="validation.{src}.{vocab_size}.spm"/' $EXP_DATA_INI
    sed -i -e 's/^tgt_test_name=.*$/tgt_test_name="validation.{tgt}.{vocab_size}.spm"/' $EXP_DATA_INI
elif [ "$DATA_TYPE" == "valA" ]; then
    sed -i -e 's/^src_test_name=.*$/src_test_name="validation.A.{src}.{vocab_size}.spm"/' $EXP_DATA_INI
    sed -i -e 's/^tgt_test_name=.*$/tgt_test_name="validation.A.{tgt}.{vocab_size}.spm"/' $EXP_DATA_INI
elif [ "$DATA_TYPE" == "valB" ]; then
    sed -i -e 's/^src_test_name=.*$/src_test_name="validation.B.{src}.{vocab_size}.spm"/' $EXP_DATA_INI
    sed -i -e 's/^tgt_test_name=.*$/tgt_test_name="validation.B.{tgt}.{vocab_size}.spm"/' $EXP_DATA_INI
elif [ "$DATA_TYPE" == "test" ]; then
    sed -i -e 's/^src_test_name=.*$/src_test_name="test.{src}.{vocab_size}.spm"/' $EXP_DATA_INI
    sed -i -e 's/^tgt_test_name=.*$/tgt_test_name="test.{tgt}.{vocab_size}.spm"/' $EXP_DATA_INI
else
    echo "Invalid data type"
    exit 1
fi

if [ "$CHECKPOINT_TYPE" == "best" ]; then
    VARIABLES_BEST=$(cat "$EXP_PATH/variables.data.best")
    sed -i -e 's~^variables=.*$~variables=\["'"$EXP_PATH/$VARIABLES_BEST"'"\]~' $EXP_DATA_INI
elif [ "$CHECKPOINT_TYPE" == "avg" ]; then
    sed -i -e 's~^variables=.*$~variables=\["'"$EXP_PATH/variables.data.avg-0"'"\]~' $EXP_DATA_INI
else
    echo "Invalid checkpoint type"
    exit 1
fi

# generate output file
set -o xtrace
./run_model.py "$EXP_INI" "$EXP_DATA_INI" --beam "$BEAM_SIZE" --kenlm "$LM_MODEL" --lm-weight "$LM_WEIGHT" --null-trail-weight "$NULL_TRAIL_WEIGHT" --nt-ratio-weight "$NT_RATIO_WEIGHT" --out "$OUT_FILE.spm"
set +o xtrace

echo "Decoding spm file"
/home/helcl/bin/spm_decode --model=$SPM_MODEL --input_format=piece --output $OUT_FILE < $OUT_FILE.spm

echo "Removing spm file"
rm "$OUT_FILE.spm"