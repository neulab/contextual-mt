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

## Pre-processing

Most of the preprocessing is done as part of the trainining.

The only thing needed is to train a bpe on the tokenized corpus. 

```bash
BPE_TOKENS=10000

python flatten_chat.py $DATA_DIR/train.json | sacremoses tokenize > /tmp/train.flat
fast learnbpe $BPE_TOKENS /tmp/train.flat > $DATA_DIR/bpecodes
fast applybpe /tmp/train.flat.$BPE_TOKENS /tmp/train.flat $DATA_DIR/bpecodes
fast getvocab /tmp/train.flat.$BPE_TOKENS > $DATA_DIR/dict.txt
rm /tmp/train.flat /tmp/train.flat.$BPE_TOKENS 
```

## Training

You can train using fairseq's training tool. Just select the `dialogue_translation` task with the approriate context sizes

```bash
fairseq-train $DATA_DIR --user-dir dialogue_mt \
    --task dialogue_translation --source-context-size $N --target-context-size $M \
    --tokenizer moses --bpe fastbpe --bpe-codes $DATA_DIR/bpecodes
    --arch transformer --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0005 --min-lr 1e-09 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout 0.2 --weight-decay 0.0 \
    --max-tokens  4096  --patience 5 --seed 42 \
    --save-dir checkpoints --no-epoch-checkpoints
```

## Inference and Evaluation


