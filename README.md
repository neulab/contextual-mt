# Dialogue MT

Implementations of context-aware models for dialogue translation on fairseq.

Currently supports:

<a href="https://arxiv.org/pdf/1708.05943.pdf"> Neural Machine Translation with Extended Context</a>

* N+M concatenation models with speaker and break tags.

## Requirements 

* [FairSeq](https://github.com/pytorch/fairseq) version >= 0.9.0
* Python version >= 3.6

## Pre-processing

Most of the preprocessing is done as part of the trainining.

The only thing needed is to train a bpe on the tokenized corpus. 
To do so, clone the subword and moses repos

```bash
git clone https://github.com/rsennrich/subword-nmt.git
BPEROOT=`pwd`/subword-nmt/subword_nmt
git clone https://github.com/moses-smt/mosesdecoder.git
TOKENIZER=`pwd`/mosesdecoder/scripts/tokenizer/tokenizer.perl
```

and train the bpe model / export the vocabulary

```bash
BPE_TOKENS=10000

python flatten_chat.py $DATA_DIR/train.json \
    | perl $TOKENIZER > /tmp/train.flat
python $BPEROOT/learn_joint_bpe_and_vocab.py \
    -i /tmp/train.flat \
    -s $BPE_TOKENS \
    -o $DATA_DIR/bpecodes \
    --write-vocabulary $DATA_DIR/dict.txt
rm /tmp/train.flat
```

## Training

You can train using fairseq's training tool. Just select the `dialogue_translation` task with the approriate context sizes

```bash
fairseq-train $DATA_DIR --user-dir dialogue_mt/tasks \
    --task dialogue_translation --source-context-size $N --target-context-size $M \
    --arch transformer --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0005 --min-lr 1e-09 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout 0.2 --weight-decay 0.0 \
    --max-tokens  4096  --patience 5 --seed 42 \
    --save-dir checkpoints --no-epoch-checkpoints
```

## Inference and Evaluation


