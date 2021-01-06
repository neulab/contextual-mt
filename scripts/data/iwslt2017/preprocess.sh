#!/bin/bash

set -euo pipefail

REPO=/home/pfernand/repos/dialogue-mt

url="https://wit3.fbk.eu/archive/2017-01-trnted/texts/de/en/de-en.tgz"
data="/projects/tir1/corpora/dialogue_mt/iwslt2017/en-de"
raw=$data/raw
prep=$data/prep-pretrained
bin=$data/bin-pretrained
src_l="en"
tgt_l="de"
src_dict=/projects/tir1/corpora/dialogue_mt/paracrawl/en-de/bin/dict.$src_l.txt
tgt_dict=/projects/tir1/corpora/dialogue_mt/paracrawl/en-de/bin/dict.$tgt_l.txt
src_spm=/projects/tir1/corpora/dialogue_mt/paracrawl/en-de/prep/spm.model
tgt_spm=/projects/tir1/corpora/dialogue_mt/paracrawl/en-de/prep/spm.model
vocab_size=32000
joint_vocab=false

mkdir -p $raw $prep $bin

archive=$raw/${url##*/}
if [ -f "$archive" ]; then
    echo "$archive already exists, skipping download and extraction..."
else
    wget -P $raw $url
    if [ -f "$archive" ]; then
        echo "$url successfully downloaded."
    else
        echo "$url not successfully downloaded."
        exit 1
    fi

    tar --strip-components=1 -C $raw -xzvf $archive
fi

echo "extract from raw data..."
rm -f $data/*.${src_l}-${tgt_l}.*
python ${REPO}/scripts/data/iwslt2017/prepare_corpus.py $raw $data -s $src_l -t $tgt_l 

if [ -z $src_spm ] || [ -z $tgt_spm ]|| [ -z $src_dict ] || [ -z $tgt_dict ]; then
    # FIXME: add option for separate vocabs
    echo "building sentencepiece model..."
    if [ "$joint_vocab" = true ]; then
        cat $data/train.${src_l}-${tgt_l}.${src_l} $data/train.${src_l}-${tgt_l}.${tgt_l} > /tmp/train.${src_l}-${tgt_l}.all
        python scripts/spm_train.py /tmp/train.${src_l}-${tgt_l}.all \
            --model-prefix $prep/spm \
            --vocab-file $prep/dict.${src_l}-${tgt_l}.txt \
            --vocab-size $vocab_size
        rm /tmp/train.${src_l}-${tgt_l}.all
        ln -s $prep/dict.${src_l}-${tgt_l}.txt $prep/dict.${src_l}.txt
        ln -s $prep/dict.${src_l}-${tgt_l}.txt $prep/dict.${tgt_l}.txt
        ln -s $prep/spm.model $prep/spm.${src_l}.model
        ln -s $prep/spm.model $prep/spm.${tgt_l}.model
    else
        for lang in $src_l $tgt_l; do
            python scripts/spm_train.py $data/train.${src_l}-${tgt_l}.${lang} \
                --model-prefix $prep/spm.${lang} \
                --vocab-file $prep/dict.${lang}.txt \
                --vocab-size $vocab_size
        done
    fi
else
    ln -sf $src_dict $prep/dict.${src_l}.txt
    ln -sf $tgt_dict $prep/dict.${tgt_l}.txt
    ln -sf $src_spm $prep/spm.${src_l}.model
    ln -sf $tgt_spm $prep/spm.${tgt_l}.model
fi


echo "applying sentencepiece model..."
for split in "train" "valid" "test"; do 
    for lang in $src_l $tgt_l; do 
        python scripts/spm_encode.py \
            --model $prep/spm.$lang.model \
                < $data/${split}.${src_l}-${tgt_l}.${lang} \
                > $prep/${split}.${src_l}-${tgt_l}.${lang}
    done
done

echo "binarizing..."
fairseq-preprocess \
    --source-lang ${src_l} --target-lang ${tgt_l} \
    --trainpref ${prep}/train.${src_l}-${tgt_l} --validpref ${prep}/valid.${src_l}-${tgt_l} --testpref ${prep}/test.${src_l}-${tgt_l} \
    --srcdict ${prep}/dict.${src_l}.txt --tgtdict ${prep}/dict.${tgt_l}.txt \
    --destdir ${bin} \
    --workers 20

cp $data/*.docids $bin