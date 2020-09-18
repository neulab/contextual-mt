DEST_DIR=../data/preprocessed/mbart_pretrained
INTER_DATA=../data/inter
MBART_DATA=$INTER_DATA/mbart_pretrained
MULTI_DATA=$INTER_DATA/multilingual
MODEL=mbart.cc25/sentence.bpe.model
DICT=mbart.cc25/dict.txt
LANGS=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

#wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.CC25.tar.gz
#tar -xzvf mbart.CC25.tar.gz

echo "copy data..."
mkdir -p $MBART_DATA
mkdir -p $DEST_DIR
cp -r $MULTI_DATA/* $MBART_DATA

#mkdir -p temp
#cp mbart.cc25/model.pt temp/

for corpus in "openSub_enru" "openSub_enfr" "openSub_ende" "openSub_enet"; do
    echo "processing ${corpus} corpus..."
    src=${corpus: -4:-2}
    tgt=${corpus: -2} 
    corpus_data=$MBART_DATA/$corpus
    pass_data=$MBART_DATA/pass/$corpus/
    
    mkdir -p $pass_data
    if [ $src == "en" ];
    then
        src_tmp="en_XX"
    elif [ $src == "ru" ];
    then
        src_tmp="ru_RU"
    elif [ $src == "fr" ];
    then
        src_tmp="fr_XX"
    elif [ $src =="de" ];
    then
        src_tmp="de_DE"
    elif [ $src == "et" ];
    then
        src_tmp="et_EE"
    fi
    
    if [ $tgt == "en" ];
    then
        tgt_tmp="en_XX"
    elif [ $tgt == "ru" ];
    then
        tgt_tmp="ru_RU"
    elif [ $tgt == "fr" ];
    then
        tgt_tmp="fr_XX"
    elif [ $tgt == "de" ];
    then
        tgt_tmp="de_DE"
    elif [ $tgt == "et" ];
    then
        tgt_tmp="et_EE"
    fi
    
    spm_encode --model=${MODEL} < $corpus_data/train.$src-$tgt.${src} > $pass_data/train.${src_tmp}
    spm_encode --model=${MODEL} < $corpus_data/train.$src-$tgt.${tgt} > $pass_data/train.${tgt_tmp}
    spm_encode --model=${MODEL} < $corpus_data/dev.$src-$tgt.${src} > $pass_data/dev.${src_tmp}
    spm_encode --model=${MODEL} < $corpus_data/dev.$src-$tgt.${tgt} > $pass_data/dev.${tgt_tmp}
    spm_encode --model=${MODEL} < $corpus_data/test.$src-$tgt.${src} > $pass_data/test.${src_tmp}
    spm_encode --model=${MODEL} < $corpus_data/test.$src-$tgt.${tgt} > $pass_data/test.${tgt_tmp}
    
    mkdir -p temp/$corpus/
    python build_vocab.py --pretrain_dict_file mbart.cc25/dict.txt  --data_dir $pass_data --dest_dir temp/$corpus/
    
    echo preprocessing ...
    
    : "fairseq-preprocess \
      --source-lang ${src_tmp} \
      --target-lang ${tgt_tmp} \
      --trainpref $pass_data/train \
      --validpref $pass_data/dev \
      --testpref $pass_data/test \
      --destdir $DEST_DIR/$corpus/ \
      --thresholdtgt 0 \
      --thresholdsrc 0 \
      --srcdict temp/$corpus/dict.txt \
      --tgtdict temp/$corpus/dict.txt \
      --workers 70"
      
    echo starting pruning ...
    python pruning_mbart.py --pretrain_dict_file mbart.cc25/dict.txt --pretrain_model_file temp/model.pt --ft_dict temp/$corpus/dict.txt --langs $LANGS --output temp/$corpus/
    
    echo ""
    
done