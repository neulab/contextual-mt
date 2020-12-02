#!/bin/bash
#
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --output=./outputs/eval_constrastive.out
#SBATCH --time=0
REPO=/home/pfernand/repos/dialogue-mt

checkpoint_dir=/projects/tir5/users/patrick/checkpoints/iwslt2017/en-de/one_to_two_pretrained_2
src_l=en
tgt_l=de

# aux files for contrastive evaluation (ContraPro)
contr_base=/home/pfernand/repos/ContraPro
contr_source=$contr_base/contrapro.text.en
contr_src_ctx=$contr_base/contrapro.context.en
contr_target=$contr_base/contrapro.text.de
contr_tgt_ctx=$contr_base/contrapro.context.de

for i in $(seq 1 10); do
    srun python $REPO/dialogue_mt/docmt_contrastive_eval.py \
        --path $checkpoint_dir \
        --checkpoint-file checkpoint$i.pt \
        --source-lang $src_l --target-lang $tgt_l \
        --source-file $contr_source \
        --src-context-file $contr_src_ctx \
        --target-file $contr_target \
        --tgt-context-file $contr_tgt_ctx
done
    