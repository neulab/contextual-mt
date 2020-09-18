model_dir=mbart.cc25
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

for corpus in "openSub_ende"; do
    src_tmp=${corpus: -4:-2}
    tgt_tmp=${corpus: -2} 
    pretrain=temp/$corpus/model.pt
    
    if [ $src_tmp == "en" ];
    then
        src="en_XX"
    elif [ $src_tmp == "ru" ];
    then
        src="ru_RU"
    elif [ $src_tmp == "fr" ];
    then
        src="fr_XX"
    elif [ $src_tmp =="de" ];
    then
        src="de_DE"
    elif [ $src_tmp == "et" ];
    then
        src="et_EE"
    fi
    
    if [ $tgt_tmp == "en" ];
    then
        tgt="en_XX"
    elif [ $tgt_tmp == "ru" ];
    then
        tgt="ru_RU"
    elif [ $tgt_tmp == "fr" ];
    then
        tgt="fr_XX"
    elif [ $tgt_tmp == "de" ];
    then
        tgt="de_DE"
    elif [ $tgt_tmp == "et" ];
    then
        tgt="et_EE"
    fi
    
    CUDA_VISIBLE_DEVICES=0,1 fairseq-train ../data/preprocessed/mbart_pretrained/$corpus\
      --encoder-normalize-before --decoder-normalize-before \
      --arch mbart_large --layernorm-embedding \
      --task translation_from_pretrained_bart \
      --source-lang $src --target-lang $tgt \
      --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
      --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
      --lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 --warmup-updates 2500 --total-num-update 40000 \
      --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
      --max-tokens 1024 --update-freq 3 \
      --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
      --save-dir checkpoint/$corpus \
      --seed 0 --log-format simple --log-interval 2 \
      --restore-file $pretrain \
      --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
      --langs $langs \
      --ddp-backend no_c10d
      
    :"fairseq-generate ../../data/preprocessed/mbart_pretrained/$corpus \
      --path $PRETRAIN \
      --task translation_from_pretrained_bart \
      --gen-subset test \
      -t $tgt -s $src \
      --bpe 'sentencepiece' --sentencepiece-model $model_dir/sentence.bpe.model \
      --sacrebleu --remove-bpe 'sentencepiece' \
      --max-sentences 32 --langs $langs > "mbart_${corpus}_result.txt" 

    #cat en_ro | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[ro_RO\]//g' |$TOKENIZER ro > en_ro.hyp
    #cat en_ro | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[ro_RO\]//g' |$TOKENIZER ro > en_ro.ref
    #sacrebleu -tok 'none' -s 'none' en_ro.ref < en_ro.hyp"

    : "model_dir=MBART_finetuned_enro # fix if you moved the checkpoint

    fairseq-generate path_2_data \
      --path $model_dir/model.pt \
      --task translation_from_pretrained_bart \
      --gen-subset test \
      -t ro_RO -s en_XX \
      --bpe 'sentencepiece' --sentencepiece-model $model_dir/sentence.bpe.model \
      --sacrebleu --remove-bpe 'sentencepiece' \
      --max-sentences 32 --langs $langs > en_ro

    cat en_ro | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[ro_RO\]//g' |$TOKENIZER ro > en_ro.hyp
    cat en_ro | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[ro_RO\]//g' |$TOKENIZER ro > en_ro.ref
    sacrebleu -tok 'none' -s 'none' en_ro.ref < en_ro.hyp
    "
done



PRETRAIN=mbart.cc25 # fix if you moved the downloaded checkpoint
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

