# Contextual  MT

Implementations of context-aware models for document-level translation tasks, used in 

1. [Measuring and Incresing Context Usage in Context-Aware Machine Translation](FIXME)

Currently supports:

* Training concatenation-based document-level machine translation models
* Training with CoWord dropout and dynamic context size
* Measuring CXMI for models 


## Requirements 

* [Fairseq](https://github.com/pytorch/fairseq) >= [add65ad](https://github.com/pytorch/fairseq/commit/add65adcc53a927f99a717d90a9672765237d937)
* [SentencePiece](https://github.com/google/sentencepiece) >= 0.1.90
* [COMET](https://github.com/Unbabel/COMET)
* Python >= 3.6

Also run 

```bash
pip install -e .
```

To have access to the entrypoints (such as the evaluation script) in your path

## Pre-processing

Preprocessing consists of training a SentencePiece model and binarizing the data
To preprocess a set of files `{train,valid,test}.{src,tgt}` run

An example prerpocessing would be

```bash
VOCAB_SIZE=32000

for lang in en fr; do
    python $REPO/scripts/spm_train.py \
        ${data_dir}/train.${lang} \
        --model-prefix ${data_dir}/prep/spm.${lang} \
        --vocab-file ${data_dir}/prep/dict.${lang}.txt \
        --vocab-size $VOCAB_SIZE
done
for split in train valid test; do
    for lang in en fr; do
        python $REPO/scripts/spm_encode.py \
            --model ${data_dir}/prep/spm.$lang.model \
                < ${data_dir}/${split}.${lang} \
                > ${data_dir}/prep/${split}.sp.${lang}
    done
done
fairseq-preprocess \
    --source-lang src --target-lang tgt \
    --trainpref ${data_dir}/prep/train.sp \
    --validpref ${data_dir}/pep/valid.sp \
    --testpref ${data_dir}/prep/test.sp \
    --srcdict ${data_dir}/prep/dict.src.txt \
    --tgtdict ${data_dir}/prep/dict.tgt.txt \
    --destdir ${data_dir}/bin
```

## Training

### Document-level translation

You can train using fairseq's training tool. Just select the `document_translation` task with the approriate context sizes.

For example, to train a model, with `N` source context size and `M` target context size, with dynamic sampling and CoWord dropout

```bash
fairseq-train \
    ${bin_dir} --user-dir $REPO/contextual_mt \
    --task document_translation \
    --source-context-size $N --target-context-size $M \
    --sample-context-size \
    --coword-dropout 0.1 \
    --log-interval 10 \
    --arch contextual_transformer --share-decoder-input-output-embed  \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
    --lr 5e-4 --lr-scheduler inverse_sqrt  --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout 0.3 --weight-decay 0.0001 \
    --max-tokens  4096 --update-freq 8 --patience 10 --seed 42 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir ${checkpoint_dir} --no-epoch-checkpoints
```

## Inference and Evaluation

You can then run inference

```bash
cp ${data_dir}/dict.*txt ${data_dir}/spm* $CHECKPOINTS_DIR

python contextual_mt/docmt_translate.py \
    --path $checkpoint_dir \
    --source-lang src --target-lang tgt \
    --source-file ${data_dir}/test.src \
    --predictions-file ${predictions_dir}/test.pred.tgt \
    --docids-file ${data_dir}/test.docids \
    --beam 5 
```

To score the predictions, a helper script is provided

```bash
python scripts/score.py ${predictions_dir}/test.pred.tgt ${data_dir}/test.tgt \
    --src ${data_dir}/test.src \
    --comet-model wmt-large-da-estimator-1719 \
    --comet-path $COMET_DIR
```

## Measuring CXMI

To measure the CXMI for a model for different context sizes, please refer to the notebook `notebooks/measuring_context_usage.ipynb`

## Contrastive evaluation

To run contrastive evaluation on ContraPro or Bawden's dataset

```bash
python contextual_mt/docmt_contrastive_eval.py \
    --source-lang src --target-lang tgt \
    --source-file $source_contr \
    --src-context-file $source_ctx_contr \
    --target-file $target_contr \
    --tgt-context-file $target_ctx_contr
```
