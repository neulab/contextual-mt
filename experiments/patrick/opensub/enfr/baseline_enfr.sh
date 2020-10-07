#!/bin/bash
#
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --output=./outputs/baseline_enfr.out
#SBATCH --time=0

EXPERIMENT_NAME=baseline_enfr
REPO=/home/pfernand/repos/dialogue-mt
DATA_DIR=/projects/tir5/users/patrick/data/opensub/enfr
CHECKPOINT_DIR=/projects/tir5/users/patrick/checkpoints/${EXPERIMENT_NAME}

srun fairseq-train \
    $DATA_DIR --user-dir $REPO/dialogue_mt \
    --task dialogue_translation \
    --bpe sentencepiece --sentencepiece-model $DATA_DIR/spm.model \
    --source-context-size 0 --target-context-size 0 \
    --log-interval 10 \
    --arch transformer --share-decoder-input-output-embed  \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
    --lr 7e-4 --lr-scheduler inverse_sqrt  --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout 0.3 --weight-decay 0.0001 \
    --max-tokens  4096 --update-freq 8 --patience 10 --seed 42 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir $CHECKPOINT_DIR --no-epoch-checkpoints

cp $DATA_DIR/dict.txt $CHECKPOINT_DIR

srun dialogue-evaluate $DATA_DIR \
    --path $CHECKPOINT_DIR \
    --split test \
    --batch-size 64 \
    --beam 5