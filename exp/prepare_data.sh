#!/usr/bin/env bash

#git clone https://github.com/moses-smt/mosesdecoder.git
#git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=20000

prep=data/inter/multilingual/pass

for corpus in "openSub_ende" "openSub_enfr" "openSub_enru" "openSub_enes"; do
    orig=data/inter/multilingual/$corpus
    src=${corpus: -4:-2}
    tgt=${corpus: -2}   
    tmp=$prep/$corpus
    
    mkdir -p $tmp    
    echo "pre-processing train, test and dev data..."
    for pair_lang in "$src-$tgt" "$tgt-$src"; do
        for l in $src $tgt; do
            for partition in "train" "dev" "test"; do
                f=$partition.$pair_lang.$l

                cat $orig/$f | \
                    perl $NORM_PUNC $l | \
                    perl $TOKENIZER -threads 8 -l $l > $prep/$f
                echo ""
            done
        done
    done
    
    TRAIN=$prep/train.all
    BPE_CODE=$tmp/code
    rm -f $TRAIN
    for l in $src $tgt; do
        cat $prep/train."$src-$tgt".$l >> $TRAIN
    done
    
    echo "learn_bpe.py on ${TRAIN}..."
    python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

    for pair_lang in "$src-$tgt" "$tgt-$src"; do
        for L in $src $tgt; do
            for f in train.$pair_lang.$L dev.$pair_lang.$L test.$pair_lang.$L; do
                echo "apply_bpe.py to ${f}..."
                python $BPEROOT/apply_bpe.py -c $BPE_CODE < $prep/$f > $tmp/$f
            done
        done
    done
    
    fairseq-preprocess --source-lang $src --target-lang $tgt \
        --trainpref $tmp/train."$src-$tgt" \
        --validpref $tmp/dev."$src-$tgt" \
        --testpref $tmp/test."$src-$tgt" \
        --joined-dictionary \
        --destdir data/preprocessed/multilingual/$corpus \
        --workers 20
    
done
    
prep=data/inter/agnostic/pass
for corpus in "openSub_ende" "openSub_enfr" "openSub_enru" "openSub_enes"; do
     orig=data/inter/agnostic/$corpus
    src=${corpus: -4:-2}
    tgt=${corpus: -2}   
    tmp=$prep/$corpus

    mkdir -p $tmp    
    echo "pre-processing train, test and dev data..."
    for l in "input" "output"; do
        for partition in "train" "dev" "test"; do
            f=$partition.$l

            cat $orig/$f | \
                perl $NORM_PUNC $l | \
                perl $TOKENIZER -threads 8 -l $l > $prep/$f
            echo ""
        done
    done

    TRAIN=$prep/train.all
    BPE_CODE=$tmp/code
    rm -f $TRAIN
    for l in "input" "output"; do
        cat $prep/train.$l >> $TRAIN
    done

    echo "learn_bpe.py on ${TRAIN}..."
    python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

    for L in "input" "output"; do
        for f in train.$L dev.$L test.$L; do
            echo "apply_bpe.py to ${f}..."
            python $BPEROOT/apply_bpe.py -c $BPE_CODE < $prep/$f > $tmp/$f
        done
    done

    TEXT=data/inter/agnostic/pass
    fairseq-preprocess --source-lang "input" --target-lang "output" \
        --trainpref $tmp/train --validpref $tmp/dev --testpref $tmp/test \
        --destdir data/preprocessed/agnostic/$corpus \
        --workers 20
done
