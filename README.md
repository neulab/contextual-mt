# Dialogue MT

Implementations of context-aware models for dialogue translation on fairseq.

Currently supports:

<a href="https://arxiv.org/pdf/1708.05943.pdf"> Neural Machine Translation with Extended Context</a>

* N+M concatenation models with speaker and break tags.

## Requirements 

* [Fairseq](https://github.com/pytorch/fairseq) >= 0.9.0
* [Sacremoses](https://github.com/alvations/sacremoses) >= 0.0.42
* [FastBPE](https://github.com/glample/fastBPE) >= 0.1.0
* Python >= 3.6

Also run 

```bash
pip install -e .
```

To have access to the entrypoints (such as the evaluation script) in your path

## Pre-processing

Most of the preprocessing is done as part of the trainining.

The only thing needed is to train the sentencepiece model

```bash
BPE_TOKENS=20000

python scripts/spm_train.py $DATA_DIR/train.json \
    --include-target \
    --model-prefix $DATA_DIR/spm \
    --vocab-file $DATA_DIR dict.txt \
    --vocab-size $BPE_TOKENS
```

## Training

You can train using fairseq's training tool. Just select the `dialogue_translation` task with the approriate context sizes

```bash
dialogue-train $DATA_DIR \
    --user-dir dialogue_mt \
    --task dialogue_translation --source-context-size $N --target-context-size $M \
    --bpe sentencepiece --sentencepiece-model $DATA_DIR/spm.model \
    --arch transformer --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
    --lr 7e-4 --lr-scheduler inverse_sqrt  --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout 0.3 --weight-decay 0.0001 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-print-samples \
    --max-tokens  4096 --update-freq 8 --patience 10 --seed 42 \
    --save-dir $CHECKPOINTS_DIR --no-epoch-checkpoints --datamap
```

## Inference and Evaluation

You can then run evaluation by running

```bash
cp $DATA_DIR/dict.txt $CHECKPOINTS_DIR
dialogue-evaluate $DATA_DIR \
    --path $CHECKPOINTS_DIR --split test \
    --batch-size 64 --beam 5 \
    --comet-model wmt-large-da-estimator-1719 \
    --comet-path $COMET_DIR 
```

Run contrastive evalution with
```bash
contrastive-evaluate $DATA_DIR --user-dir $REPO/dialogue_mt --dataset-impl mmap\
    --source-context-size $N --target-context-size $M \
    --task dialogue_translation --path $REPO/checkpoints/$EXPERIMENT_NAME/checkpoint_best.pt --max-tokens 4096 \
    --bpe sentencepiece --sentencepiece-model $DATA_DIR/spm.model \
    --prefix-size -1 --output $REPO/contra/$EXPERIMENT_NAME.txt --contra $CONTRA_DIR
```
which will produce the model's scores for each contrastive sample. 