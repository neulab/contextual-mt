#!/bin/bash
#
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --output=./outputs/iwslt2017/en-de/two_to_two_pretrained_drop01_1
#SBATCH --time=0

REPO=/home/pfernand/repos/dialogue-mt
COMET_DIR=/projects/tir1/users/pfernand/comet/

# define basic variables for experiment
experiment_name=two_to_two_pretrained_1
data_dir=/projects/tir1/corpora/dialogue_mt/iwslt2017/en-de
bin_dir=${data_dir}/bin-pretrained
prep_dir=${data_dir}/prep-pretrained
checkpoint_dir=/projects/tir5/users/patrick/checkpoints/iwslt2017/en-de/${experiment_name}
pretrained_checkpoint=/projects/tir5/users/patrick/checkpoints/paracrawl/pretrain_large/checkpoint_best.pt
predictions_dir=./predictions/iwslt2017/en-de/${experiment_name}
src_l="en"
tgt_l="de"
seed=1

# aux files for evaluating BLEU
test_ref_file=${data_dir}/test.${src_l}-${tgt_l}.${tgt_l}
test_src_file=${data_dir}/test.${src_l}-${tgt_l}.${src_l}
test_docids_file=${data_dir}/test.${src_l}-${tgt_l}.docids
valid_ref_file=${data_dir}/valid.${src_l}-${tgt_l}.${tgt_l}
valid_src_file=${data_dir}/valid.${src_l}-${tgt_l}.${src_l}
valid_docids_file=${data_dir}/valid.${src_l}-${tgt_l}.docids

# aux files for contrastive evaluation (ContraPro)
contr_base=/home/pfernand/repos/ContraPro
contr_source=$contr_base/contrapro.text.en
contr_src_ctx=$contr_base/contrapro.context.en
contr_target=$contr_base/contrapro.text.de
contr_tgt_ctx=$contr_base/contrapro.context.de

mkdir -p $predictions_dir

srun fairseq-train \
    $bin_dir --user-dir $REPO/dialogue_mt \
    --fp16 \
    --task document_translation \
    --source-context-size 1 --target-context-size 1 \
    --source-dropout 0.1 \
    --log-interval 10 \
    --finetune-from-model $pretrained_checkpoint \
    --arch contextual_transformer_big --share-decoder-input-output-embed  \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
    --lr 1e-4 --lr-scheduler inverse_sqrt  --warmup-updates 1000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout 0.3 --weight-decay 0.0001 \
    --max-tokens  2048 --update-freq 8 --patience 10 --seed 42 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir $checkpoint_dir --no-epoch-checkpoints

cp $bin_dir/dict* $checkpoint_dir
cp $prep_dir/spm* $checkpoint_dir 
    
srun python $REPO/dialogue_mt/docmt_translate.py \
    --path $checkpoint_dir \
    --source-file $valid_src_file \
    --reference-file $valid_ref_file \
    --predictions-file $predictions_dir/valid.${experiment_name}.$src_l-$tgt_l.$tgt_l \
    --docids-file $valid_docids_file \
    --source-lang $src_l --target-lang $tgt_l \
    --beam 5 

echo "evaluating on valid..."
srun python $REPO/scripts/score.py \
    $predictions_dir/valid.$experiment_name.$src_l-$tgt_l.$tgt_l $valid_ref_file \
    --src $valid_src_file \
    --comet-model wmt-large-da-estimator-1719 --comet-path $COMET_DIR

srun python $REPO/dialogue_mt/docmt_translate.py \
    --path $checkpoint_dir \
    --source-file $valid_src_file \
    --reference-file $valid_ref_file \
    --predictions-file $predictions_dir/valid.${experiment_name}.gold.$src_l-$tgt_l.$tgt_l \
    --docids-file $valid_docids_file \
    --source-lang $src_l --target-lang $tgt_l \
    --beam 5 --gold-target-context

echo "evaluating on valid (gold)..."
srun python $REPO/scripts/score.py \
    $predictions_dir/valid.$experiment_name.gold.$src_l-$tgt_l.$tgt_l $valid_ref_file \
    --src $valid_src_file \
    --comet-model wmt-large-da-estimator-1719 --comet-path $COMET_DIR

srun python $REPO/dialogue_mt/docmt_translate.py \
    --path $checkpoint_dir \
    --source-file $test_src_file \
    --reference-file $test_ref_file \
    --predictions-file $predictions_dir/test.${experiment_name}.$src_l-$tgt_l.$tgt_l \
    --docids-file $test_docids_file \
    --source-lang $src_l --target-lang $tgt_l \
    --beam 5 

echo "evaluating on test..."
srun python $REPO/scripts/score.py \
    $predictions_dir/test.$experiment_name.$src_l-$tgt_l.$tgt_l $test_ref_file \
    --src $test_src_file \
    --comet-model wmt-large-da-estimator-1719 --comet-path $COMET_DIR

srun python $REPO/dialogue_mt/docmt_translate.py \
    --path $checkpoint_dir \
    --source-file $test_src_file \
    --reference-file $test_ref_file \
    --predictions-file $predictions_dir/test.${experiment_name}.gold.$src_l-$tgt_l.$tgt_l \
    --docids-file $test_docids_file \
    --source-lang $src_l --target-lang $tgt_l \
    --beam 5 --gold-target-context

echo "evaluating on test (gold)..."
srun python $REPO/scripts/score.py $predictions_dir/test.$experiment_name.gold.$src_l-$tgt_l.$tgt_l $test_ref_file \
    --src $test_src_file \
    --comet-model wmt-large-da-estimator-1719 --comet-path $COMET_DIR

echo "evaluating on contrastive set..."
srun python $REPO/dialogue_mt/docmt_contrastive_eval.py \
    --path $checkpoint_dir \
    --source-lang $src_l --target-lang $tgt_l \
    --source-file $contr_source \
    --src-context-file $contr_src_ctx \
    --target-file $contr_target \
    --tgt-context-file $contr_tgt_ctx