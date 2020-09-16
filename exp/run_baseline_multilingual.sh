
for corpus in "openSub_enru" "openSub_enfr" "openSub_enes" "openSub_ende"; do
    src=${corpus: -4:-2}
    tgt=${corpus: -2} 
    
    mkdir -p checkpoints/multilingual_transformer
    CUDA_VISIBLE_DEVICES=1 fairseq-train data/preprocessed/multilingual/$corpus/ \
        --max-epoch 80 \
        --ddp-backend=no_c10d \
        --task multilingual_translation --lang-pairs "$src-$tgt","$tgt-$src" \
        --arch multilingual_transformer_iwslt_de_en \
        --share-decoder-input-output-embed \
        --share-encoders --share-decoders \
        --optimizer adam --adam-betas '(0.9, 0.98)' \
        --lr 5e-4 --lr-scheduler inverse_sqrt --clip-norm 0.0 \
        --warmup-updates 4000 \
        --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
        --dropout 0.3 --weight-decay 0.0001 --label-smoothing 0.1 \
        --save-dir checkpoint/multilingual/$corpus \
        --max-tokens 4096 \
        --update-freq 8 \
        --seed 0 \
        --patience 10

    CUDA_VISIBLE_DEVICES=1 fairseq-generate data/preprocessed/multilingual/$corpus \
        --task multilingual_translation --lang-pairs "$src-$tgt","$tgt-$src" --source-lang $src --target-lang $tgt \
        --path checkpoint/multilingual/$corpus/checkpoint_best.pt \
        --batch-size 128 --beam 5 --remove-bpe > "multi_${corpus}_result.txt" 

done


