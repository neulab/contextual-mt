#!/usr/bin/env bash
#
# downloads, extracts, preprocess and binarizes paracrawl
# warning do not run in head node, use with 
# `--mem=64000 --cpus-per-task=10 --time=0`

set -eu

REPO=/home/pfernand/repos/dialogue-mt

SCRIPTS=~/repos/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
FASTTEXT_MODEL=~/lid.176.bin

THREADS=10
VALID_SIZE=2000
TEST_SIZE=2000

url="https://s3.amazonaws.com/web-language-models/paracrawl/release7.1/en-de.txt.gz"
data="/projects/tir1/corpora/dialogue_mt/paracrawl/en-de"
raw=$data/raw
tmp=$data/tmp
prep=$data/prep
bin=$data/bin
src_lang="en"
tgt_lang="de"
vocab_size=32000
vocab_sample_size=20000000

mkdir -p $raw $prep $bin $tmp

# archive=$raw/${url##*/}
# data_file=${archive%.gz}
# if [ -f "$archive" ]; then
#     echo "$data_file exists, skipping download and extraction..."
# else
#     wget -P $raw $url
#     if [ -f "$archive" ]; then
#         echo "$url successfully downloaded."
#     else
#         echo "$url not successfully downloaded."
#         exit 1
#     fi
#     echo "extracting archive..."
#     gunzip $archive
# fi
#  
# # echo "splitting data..."
# # total_size=`wc -l $data_file | cut -f1 -d ' '`
# # eval_size=$(( $VALID_SIZE + $TEST_SIZE ))
# # train_size=$(( $total_size - $eval_size ))
# # shuf $data_file > $tmp/data.rd.txt
# # head -$train_size $tmp/data.rd.txt > $tmp/train.txt
# # tail -$eval_size $tmp/data.rd.txt | head -$VALID_SIZE > $tmp/valid.txt
# # tail -$eval_size $tmp/data.rd.txt | tail -$TEST_SIZE > $tmp/test.txt
# # for split in train valid test; do
# #     cat $tmp/$split.txt | cut -f1 > $tmp/${split}.${src_lang}
# #     cat $tmp/$split.txt | cut -f2 > $tmp/${split}.${tgt_lang}
# # done
# # 
# # echo "cleaning data..."
# # for split in train valid test; do
# #     python $REPO/scripts/data/paracrawl/clean_corpus.py $tmp/$split $tmp/$split.cln \
# #             --source-lang $src_lang --target-lang $tgt_lang \
# #             --fasttext-model $FASTTEXT_MODEL
# # done
# 
# echo "learning sentecepiece model..."
# cat $tmp/train.cln.${src_lang} $tmp/train.cln.${tgt_lang} > $tmp/train.cln.full
# python scripts/spm_train.py $tmp/train.cln.full \
#     --model-prefix $prep/spm --vocab-size $vocab_size \
#     --vocab-sample-size $vocab_sample_size \
#     --vocab-file $prep/dict.${src_lang}-${tgt_lang}.txt
# 
# 
# echo "applying sentecepiece model..."
# python scripts/spm_encode.py --model $prep/spm.model < $tmp/train.${src_lang} > $prep/train.sp.${src_lang} &
# python scripts/spm_encode.py --model $prep/spm.model < $tmp/train.${tgt_lang} > $prep/train.sp.${tgt_lang} &
# wait
# 
# for split in valid test; do
#     python scripts/spm_encode.py --model $prep/spm.model < $tmp/$split.${src_lang} > $prep/$split.sp.${src_lang} &
#     python scripts/spm_encode.py --model $prep/spm.model < $tmp/$split.${tgt_lang} > $prep/$split.sp.${tgt_lang} &
#     wait
# done
# 
# rm -r $bin/*

echo "binarizing..."
fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang \
    --trainpref $prep/train.sp --validpref $prep/valid.sp --testpref $prep/test.sp \
    --srcdict $prep/dict.${src_lang}-${tgt_lang}.txt --tgtdict $prep/dict.${src_lang}-${tgt_lang}.txt \
    --destdir $bin \
    --workers $THREADS

rm -r $tmp