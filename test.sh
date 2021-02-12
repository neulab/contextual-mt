#!/bin/bash
#
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=0
#SBATCH --output=./logs/test.out
conda init bash
source ~/.bashrc
conda activate venv
exp_name=test
REPO=/projects/tir4/users/kayoy/contextual-mt
bin_dir=/projects/tir4/users/kayoy/attention-regularization/dialogue-mt/test_data/bin
checkpoint_dir=/projects/tir4/users/kayoy/attention-regularization/dialogue-mt/checkpoints/${exp_name}
N=5
M=5
fairseq-train \
    ${bin_dir} --user-dir $REPO/contextual_mt \
    --task attention_regularization \
    --source-context-size $N --target-context-size $M \
    --source-lang en --target-lang fr \
    --log-interval 100 \
    --arch attn_reg_transformer --share-decoder-input-output-embed  \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
    --lr 5e-4 --lr-scheduler inverse_sqrt  --warmup-updates 4000 \
    --criterion attention_loss --label-smoothing 0.1 --dropout 0.3 --weight-decay 0.0001 \
    --regularize-heads 0 --regularize-attention enc --highlight-sample 0.2 --kl-lambda 10 \
    --max-tokens  4096 --max-tokens-valid 1024 --update-freq 8 --seed 42 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe sentencepiece \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir ${checkpoint_dir} --no-epoch-checkpoints --save-interval-updates 2000 --wandb-project ${exp_name}