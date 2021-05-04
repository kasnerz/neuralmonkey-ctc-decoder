#!/bin/bash

# CPU
source /lnet/spec/work/people/kasner/virtualenv/tensorflow-1.10-cpu/bin/activate

# GPU
# source /lnet/spec/work/people/kasner/virtualenv/tensorflow-1.5-gpu/bin/activate

set -e

usage() { echo "Usage: $0 -s <en|de|ro|cs> -t <en|de|ro|cs> -c <avg|best> -d <val|test> -b <beam_size:int> [-l <wl:float>] [-n <wt:float>] [-r <wr:float>]" 1>&2; exit 1; }

if [ "$#" -lt 7 ]; then
    usage
fi

while getopts ":s:t:c:d:b:l:n:r:" o; do
    case "${o}" in
        s)
            LANG1=${OPTARG}
            echo "LANG1=$LANG1"
            ((LANG1 == "en" || LANG1 == "de" || LANG1 == "ro" || LANG1 == "cs")) || usage
            ;;
        t)
            LANG2=${OPTARG}
            echo "LANG2=$LANG2"
            ((LANG2 == "en" || LANG2 == "de" || LANG2 == "ro" || LANG2 == "cs")) || usage
            ;;
        c)
            CHECKPOINT_TYPE=${OPTARG}
            echo "CHECKPOINT_TYPE=$CHECKPOINT_TYPE"
            ((CHECKPOINT_TYPE == "avg" || CHECKPOINT_TYPE == "best")) || usage
            ;;
        d)
            DATASET_TYPE=${OPTARG}
            echo "DATASET_TYPE=$DATASET_TYPE"
            ((DATASET_TYPE == "val" || DATASET_TYPE == "test")) || usage
            ;;
        b)
            BEAM_SIZE=${OPTARG}
            echo "BEAM_SIZE=$BEAM_SIZE"
            [ $BEAM_SIZE -gt 0 ] || usage
            ;;
        l)
            LM_SCORE_DEFAULT=${OPTARG}
            echo "LM_SCORE_DEFAULT=$LM_SCORE_DEFAULT"
            ;;
        n)
            NULL_TRAILING_DEFAULT=${OPTARG}
            echo "NULL_TRAILING_DEFAULT=$NULL_TRAILING_DEFAULT"
            ;;
        r)
            NULL_TOKEN_RATIO_DEFAULT=${OPTARG}
            echo "NULL_TOKEN_RATIO_DEFAULT=$NULL_TOKEN_RATIO_DEFAULT"
            ;;
        *)
            usage
            ;;
    esac
done

# ---------------------
# VARIABLES
# ---------------------
MODEL_TYPE="shared-pos-b10"
VOCAB_SIZE=50000
LM_NGRAMS=5

EXP_FILENAME="$LANG1$LANG2-san_ctc-kd-$MODEL_TYPE-v$VOCAB_SIZE"
EXP_PATH="experiments/knowledge_distillation/$EXP_FILENAME"


EXP_ORIG_INI="$EXP_PATH/experiment.ini"
EXP_ORIG_DATA_INI="experiments/knowledge_distillation/san_kombo_ctc_data.ini"

LM_MODEL="../lm/news.$LANG2.$LANG1$LANG2.$VOCAB_SIZE.$LM_NGRAMS.bin"
SPM_MODEL="../spm/models/sp.$LANG1$LANG2.$VOCAB_SIZE.model"

TMP="./tmp"
INI_SUFFIX="$CHECKPOINT_TYPE-$DATASET_TYPE-b$BEAM_SIZE-ld$LM_SCORE_DEFAULT-td$NULL_TRAILING_DEFAULT-rd$NULL_TOKEN_RATIO_DEFAULT"
EXP_INI="$TMP/$EXP_FILENAME.$INI_SUFFIX.ini"
EXP_DATA_INI="$TMP/$EXP_FILENAME-data.$INI_SUFFIX.ini"


# ---------------------
# UPDATE CONFIG FILES
# ---------------------
# do not modify original files
cp $EXP_ORIG_INI $EXP_INI
cp $EXP_ORIG_DATA_INI $EXP_DATA_INI

# languages
sed -i -e 's/^src=.*$/src="'"$LANG1"'"/' $EXP_DATA_INI
sed -i -e 's/^tgt=.*$/tgt="'"$LANG2"'"/' $EXP_DATA_INI

# first half of the validation dataset for training the beam
sed -i -e 's/^src_test_name=.*$/src_test_name="validation.A.{src}.{vocab_size}.spm"/' $EXP_DATA_INI
sed -i -e 's/^tgt_test_name=.*$/tgt_test_name="validation.A.{tgt}.{vocab_size}.spm"/' $EXP_DATA_INI

# best / averaged checkpoint
if [ "$CHECKPOINT_TYPE" == "best" ]; then
    VARIABLES_BEST=$(cat "$EXP_PATH/variables.data.best")
    sed -i -e 's~^variables=.*$~variables=\["'"$EXP_PATH/$VARIABLES_BEST"'"\]~' $EXP_DATA_INI
else
    sed -i -e 's~^variables=.*$~variables=\["'"$EXP_PATH/variables.data.avg-0"'"\]~' $EXP_DATA_INI
fi

sed -i -e 's/^vocab_size.*$/vocab_size='"$VOCAB_SIZE"'/' $EXP_DATA_INI


# datasets for sacreBLEU
DEV_SET="wmt13"
TEST_SET="wmt14"

if [ "$LANG1" == "ro" -o "$LANG2" == "ro" ]; then
    DEV_SET="wmt16/dev"
    TEST_SET="wmt16"
elif [ "$LANG1" == "cs" -o "$LANG2" == "cs" ]; then
    TEST_SET="wmt18"
fi

echo "Datasets for langpair $LANG1-$LANG2:"
echo "validation: $DEV_SET"
echo "test: $TEST_SET"

# ---------------------
# TRAIN WEIGHTS
# ---------------------
# run training only if there is anything to train
if [ ! -z "$LM_SCORE_DEFAULT" ] || [ ! -z "$NULL_TRAILING_DEFAULT" ] || [ ! -z "$NULL_TOKEN_RATIO_DEFAULT" ]; then

    # train perceptron
    echo "---------------------------------"
    echo "Training beam search weights"
    echo "---------------------------------"

    WEIGHTS_PREFIX="$TMP/$EXP_FILENAME.$INI_SUFFIX.weights.ckp"
    TRAIN_ARGS="$EXP_INI $EXP_DATA_INI --beam $BEAM_SIZE --kenlm $LM_MODEL --prefix $WEIGHTS_PREFIX"

    if [ ! -z "$LM_SCORE_DEFAULT" ]; then
        TRAIN_ARGS+=" --lm-weight $LM_SCORE_DEFAULT"
    fi
    if [ ! -z "$NULL_TRAILING_DEFAULT" ]; then
        TRAIN_ARGS+=" --null-trail-weight $NULL_TRAILING_DEFAULT"
    fi
    if [ ! -z "$NULL_TOKEN_RATIO_DEFAULT" ]; then
        TRAIN_ARGS+=" --nt-ratio-weight $NULL_TOKEN_RATIO_DEFAULT"
    fi

    set -o xtrace
    ./train_model.py $TRAIN_ARGS
    set +o xtrace


    # validation
    CHECKPOINT_BEST=""
    CHECKPOINT_BEST_SCORE=0.0

    CNT=`ls -1 $WEIGHTS_PREFIX.*[!out] | wc -l`
    echo "Checkpoints to validate: $CNT"

    for FILE in $WEIGHTS_PREFIX.*[!out]; do
        source "$FILE"
        echo "---------------------------------"
        echo "Validating $FILE"
        echo "---------------------------------"
        cat "$FILE"

        TMP_FILE="$FILE.out"

        # second half of the validation dataset for validation the weights
        sed -i -e 's/^src_test_name=.*$/src_test_name="validation.B.{src}.{vocab_size}.spm"/' $EXP_DATA_INI
        sed -i -e 's/^tgt_test_name=.*$/tgt_test_name="validation.B.{tgt}.{vocab_size}.spm"/' $EXP_DATA_INI

        RUN_ARGS="$EXP_INI $EXP_DATA_INI --beam $BEAM_SIZE --kenlm $LM_MODEL --out $TMP_FILE.spm"

        if [ ! -z "$LM_SCORE_DEFAULT" ]; then
            RUN_ARGS+=" --lm-weight $LM_SCORE"
        fi
        if [ ! -z "$NULL_TRAILING_DEFAULT" ]; then
            RUN_ARGS+=" --null-trail-weight $NULL_TRAILING"
        fi
        if [ ! -z "$NULL_TOKEN_RATIO_DEFAULT" ]; then
            RUN_ARGS+=" --nt-ratio-weight $NULL_TOKEN_RATIO"
        fi

        set -o xtrace
        ./run_model.py $RUN_ARGS
        set +o xtrace

        echo "Decoding spm file"
        /home/helcl/bin/spm_decode --model="$SPM_MODEL" --input_format=piece --output "$TMP_FILE" < "$TMP_FILE.spm"

        SCORE=`sacrebleu "data/$LANG1$LANG2/validation.B.$LANG2.$VOCAB_SIZE" --input "$TMP_FILE" --score-only --width 2 -l "$LANG1-$LANG2"`
        rm $TMP_FILE*

        echo "BLEU score on validation set: $SCORE"

        if  (( $(echo "$SCORE > $CHECKPOINT_BEST_SCORE" | bc -l) )); then
            CHECKPOINT_BEST_SCORE="$SCORE"
            CHECKPOINT_BEST="$FILE"
        fi
    done

    # load the weights with best validation score
    echo "---------------------------------"
    echo "Loading weights"
    echo "---------------------------------"
    echo $CHECKPOINT_BEST
    cat "$CHECKPOINT_BEST"
    source "$CHECKPOINT_BEST"

    rm $WEIGHTS_PREFIX.*
fi

SUFFIX="$CHECKPOINT_TYPE-$DATASET_TYPE-b$BEAM_SIZE-wl$LM_SCORE-wt$NULL_TRAILING-wr$NULL_TOKEN_RATIO"
OUT_FILE="results/kd/$EXP_FILENAME-$SUFFIX"

# ---------------------
# GENERATE OUTPUT
# ---------------------
# generate output file
echo "---------------------------------"
echo "Generating output"
echo "---------------------------------"


# validation / test dataset
if [ "$DATASET_TYPE" == "val" ]; then
    sed -i -e 's/^src_test_name=.*$/src_test_name="validation.{src}.{vocab_size}.spm"/' $EXP_DATA_INI
    sed -i -e 's/^tgt_test_name=.*$/tgt_test_name="validation.{tgt}.{vocab_size}.spm"/' $EXP_DATA_INI
else
    sed -i -e 's/^src_test_name=.*$/src_test_name="test.{src}.{vocab_size}.spm"/' $EXP_DATA_INI
    sed -i -e 's/^tgt_test_name=.*$/tgt_test_name="test.{tgt}.{vocab_size}.spm"/' $EXP_DATA_INI
fi

RUN_ARGS="$EXP_INI $EXP_DATA_INI --beam $BEAM_SIZE --kenlm $LM_MODEL --out $OUT_FILE.spm"

if [ ! -z "$LM_SCORE_DEFAULT" ]; then
    RUN_ARGS+=" --lm-weight $LM_SCORE"
fi
if [ ! -z "$NULL_TRAILING_DEFAULT" ]; then
    RUN_ARGS+=" --null-trail-weight $NULL_TRAILING"
fi
if [ ! -z "$NULL_TOKEN_RATIO_DEFAULT" ]; then
    RUN_ARGS+=" --nt-ratio-weight $NULL_TOKEN_RATIO"
fi

set -o xtrace
./run_model.py $RUN_ARGS
set +o xtrace

echo "Decoding spm file"
../spm/spm_decode --model="$SPM_MODEL" --input_format=piece --output "$OUT_FILE" < "$OUT_FILE.spm"

echo "Removing spm file"
rm "$OUT_FILE.spm"

if [ "$DATASET_TYPE" == "val" ]; then
    sacrebleu --input "$OUT_FILE" --width 2 --test-set "$DEV_SET" -l "$LANG1-$LANG2"
else
    sacrebleu --input "$OUT_FILE" --width 2 --test-set "$TEST_SET" -l "$LANG1-$LANG2"
fi


