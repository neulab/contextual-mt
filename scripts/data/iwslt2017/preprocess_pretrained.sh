#!/bin/bash

set -euo pipefail

# Replace appropriate variables for paths and languages

REPO=/home/pfernand/repos/contextual-mt

url="https://wit3.fbk.eu/archive/2017-01-trnted/texts/fr/en/en-fr.tgz"
data="/projects/tir1/corpora/dialogue_mt/iwslt2017/en-fr"
raw=$data/raw
prep=$data/prep-pretrained
bin=$data/bin-pretrained
src_l="en"
tgt_l="fr"
src_dict="/projects/tir5/users/patrick/data/paracrawl/en-fr/bin/dict.$src_l.txt"
tgt_dict="/projects/tir5/users/patrick/data/paracrawl/en-fr/bin/dict.$tgt_l.txt"
src_spm="/projects/tir5/users/patrick/data/paracrawl/en-fr/prep/spm.model"
tgt_spm="/projects/tir5/users/patrick/data/paracrawl/en-fr/prep/spm.model"

mkdir -p $raw $prep $bin

archive=$raw/${url##*/}
echo $archive
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

ln -sf $src_dict $prep/dict.${src_l}.txt
ln -sf $tgt_dict $prep/dict.${tgt_l}.txt
ln -sf $src_spm $prep/spm.${src_l}.model
ln -sf $tgt_spm $prep/spm.${tgt_l}.model


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