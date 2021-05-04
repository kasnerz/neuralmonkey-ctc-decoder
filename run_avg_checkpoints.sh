#$ -l gpu=1
#$ -q gpu-ms.q -pty yes -S /bin/bash -m n -cwd
#$ -N avg_checkpoints -o avg_checkpoints.out -e avg_checkpoints.out

# source /lnet/spec/work/people/kasner/venv/bin/activate

EXP_DIR=experiments/knowledge_distillation

declare -a EXPS=(
    "csen-san_ctc-kd-shared-pos-b10-v50000"
    "deen-san_ctc-kd-shared-pos-b10-v50000"
    "roen-san_ctc-kd-shared-pos-b10-v50000"
    "enro-san_ctc-kd-shared-pos-b10-v50000"
    "encs-san_ctc-kd-shared-pos-b10-v50000"
    "ende-san_ctc-kd-shared-pos-b10-v50000"
    )

for EXP in "${EXPS[@]}"; do
    ./neuralmonkey/scripts/avg_checkpoints.py $EXP_DIR/$EXP/variables.data.{0..4} $EXP_DIR/$EXP/variables.data.avg
done
