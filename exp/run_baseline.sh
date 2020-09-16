
for corpus in "openSub_enru" "openSub_enfr" "openSub_enes" "openSub_ende"; do
    CUDA_VISIBLE_DEVICES=1 fairseq-train \
        data/preprocessed/$corpus \
        --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
        --dropout 0.3 --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --max-tokens 4096  --update-freq 8 --patience 10  \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok moses --seed 0 \
        --eval-bleu-print-samples \
        --eval-bleu-remove-bpe \
        --max-epoch 55 \
        --patience 6 \
        --save-dir checkpoint/$corpus \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
        
    CUDA_VISIBLE_DEVICES=1 fairseq-generate data/preprocessed/$corpus \
        --path checkpoint/$corpus/checkpoint_best.pt \
        --batch-size 128 --beam 5 --remove-bpe > "${corpus}_result.txt"
done