import "submitters.tape"

task GetData 
    < data_dir=@
    > train_src
    > train_tgt
    > train_docids
    > valid_src
    > valid_tgt
    > valid_docids
    > test_src
    > test_tgt
    > test_docids
    :: .submitter=@ .mem=8000 .gpus=0 .cpus=1
    :: src_lang=@
    :: tgt_lang=@
    :: repo=@
{
    # copy data files to the ducttape filesystem
    cp $data_dir/train.${src_lang} $train_src
    cp $data_dir/train.${tgt_lang} $train_tgt
    cp $data_dir/train.docids $train_docids
    cp $data_dir/dev.${src_lang} $valid_src
    cp $data_dir/dev.${tgt_lang} $valid_tgt
    cp $data_dir/dev.docids $valid_docids
    cp $data_dir/test.${src_lang} $test_src
    cp $data_dir/test.${tgt_lang} $test_tgt
    cp $data_dir/test.docids $test_docids
}

task TrainSentencePiece
    < train_src=@GetData
    < train_tgt=@GetData
    > src_spm 
    > tgt_spm
    > src_vocab
    > tgt_vocab
    :: .submitter=@ .mem=8000 .gpus=0 .cpus=1
    :: repo=@
    :: joint_vocab=false
    :: vocab_size=20000
{
    if [ "$joint_vocab" = true ]; then
        cat $train_src $train_tgt > train.all 
        python $repo/scripts/spm_train.py train.all \
            --model-prefix sp_model \
            --vocab-file vocab \
            --vocab-size $vocab_size
        ln -s vocab $src_vocab
        ln -s vocab $tgt_vocab
        ln -s sp_model.model $src_spm
        ln -s sp_model.model $tgt_spm
    else
        python $repo/scripts/spm_train.py $train_src \
            --model-prefix sp_model \
            --vocab-file $src_vocab \
            --vocab-size $vocab_size
        mv sp_model.model $src_spm
        python $repo/scripts/spm_train.py $train_tgt \
            --model-prefix sp_model \
            --vocab-file $tgt_vocab \
            --vocab-size $vocab_size
        mv sp_model.model $tgt_spm
    fi
}

task ApplySentencePiece
    < raw_src=(
        Split:
            train=$train_src@GetData
            valid=$valid_src@GetData
            test=$test_src@GetData
        )
    < raw_tgt=(
        Split:
            train=$train_tgt@GetData
            valid=$valid_tgt@GetData
            test=$test_tgt@GetData
        )
    < src_spm=@TrainSentencePiece
    < tgt_spm=@TrainSentencePiece
    > prep_src
    > prep_tgt
    :: .submitter=@ .mem=8000 .gpus=0 .cpus=1
    :: repo=@
{
    python $repo/scripts/spm_encode.py \
                --model $src_spm \
                    < $raw_src \
                    > $prep_src
    python $repo/scripts/spm_encode.py \
                --model $tgt_spm \
                    < $raw_tgt \
                    > $prep_tgt
}

task Binarize
    < train_src=$prep_src@ApplySentencePiece[Split:train]
    < train_tgt=$prep_tgt@ApplySentencePiece[Split:train]
    < train_docids=@GetData
    < valid_src=$prep_src@ApplySentencePiece[Split:valid]
    < valid_tgt=$prep_tgt@ApplySentencePiece[Split:valid]
    < valid_docids=@GetData
    < test_src=$prep_src@ApplySentencePiece[Split:test]
    < test_tgt=$prep_tgt@ApplySentencePiece[Split:test]
    < test_docids=@GetData
    < src_spm=@TrainSentencePiece
    < tgt_spm=@TrainSentencePiece
    < src_vocab=@TrainSentencePiece
    < tgt_vocab=@TrainSentencePiece
    > bin_dir
    :: .submitter=@ .mem=8000 .gpus=0 .cpus=10
    :: src_lang=@
    :: tgt_lang=@
{
    ln -s $train_src train.$src_lang 
    ln -s $train_tgt train.$tgt_lang 
    ln -s $valid_src valid.$src_lang 
    ln -s $valid_tgt valid.$tgt_lang 
    ln -s $test_src test.$src_lang 
    ln -s $test_tgt test.$tgt_lang 

    fairseq-preprocess \
        --source-lang $src_lang --target-lang $tgt_lang \
        --trainpref train --validpref valid --testpref test \
        --srcdict $src_vocab --tgtdict $tgt_vocab \
        --destdir $bin_dir \
        --workers 10
    
    ln -s $train_docids $bin_dir/train.${src_lang}-${tgt_lang}.docids
    ln -s $valid_docids $bin_dir/valid.${src_lang}-${tgt_lang}.docids
    ln -s $test_docids $bin_dir/test.${src_lang}-${tgt_lang}.docids
    ln -s $src_spm $bin_dir/spm.${src_lang}.model
    ln -s $tgt_spm $bin_dir/spm.${tgt_lang}.model
}

task TrainModel
    < bin_dir=@Binarize
    > checkpoint_dir
    :: .submitter=@ .mem=16000 .gpus=1 .cpus=2
    :: src_lang=@
    :: tgt_lang=@
    :: N=@
    :: M=@
    :: multi_encoder=@
    :: repo=@
    :: seed=@
    
{
    fairseq-train \
        $bin_dir --user-dir $repo/contextual_mt \
        --fp16 \
        --task document_translation \
        --source-context-size $N --target-context-size $M \
        --sample-context-size \
        --coword-dropout 0.2 \
        $([ "$multi_encoder" = true ] && echo "--multi-encoder" || echo "") \
        --log-interval 10 \
        --arch contextual_transformer_iwslt --share-decoder-input-output-embed  \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
        --lr 5e-4 --lr-scheduler inverse_sqrt  --warmup-updates 4000 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout 0.3 --weight-decay 0.0001 \
        --max-tokens  2048 --update-freq 8 --patience 10 --seed 42 \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-remove-bpe sentencepiece \
        --eval-bleu-print-samples \
        --save-dir $checkpoint_dir --no-epoch-checkpoints \
        --seed $seed

    cp $bin_dir/dict.* $bin_dir/spm.* $checkpoint_dir
}

task GeneratePredictions
    < checkpoint_dir=@TrainModel
    < test_src=@GetData
    < test_docids=@GetData
    > test_pred
    :: .submitter=@ .mem=16000 .gpus=1 .cpus=2
    :: src_lang=@
    :: tgt_lang=@
    :: decode_context_size=@
    :: gold_context=@
    :: repo=@
{
    python $repo/contextual_mt/docmt_translate.py \
        --path $checkpoint_dir \
        --source-file $test_src \
        --predictions-file $test_pred \
        $([ "$gold_context" = true ] && echo "--gold-target-context" || echo "") \
        $([ -z "$decode_context_size" ] && echo "--target-context-size $decode_context_size" || echo "") \
        --reference-file $test_tgt \
        --docids-file $test_docids \
        --source-lang $src_lang --target-lang $tgt_lang \
        --beam 5
}

task ScorePredictions
    < test_pred=@GeneratePredictions
    < test_src=@GetData
    < test_tgt=@GetData
    > score
    :: .submitter=@ .mem=16000 .gpus=1 .cpus=2
    :: repo=@
    :: comet_dir=@
{
    python $repo/scripts/score.py $test_pred $test_tgt \
        --src $test_pred \
        --comet-model wmt-large-da-estimator-1719 --comet-path $comet_dir > $score
}

task MeasureCXMI
    < test_src=@GetData
    < test_tgt=@GetData
    < test_docids=@GetData
    < checkpoint_dir=@TrainModel
    > cxmi_results
    :: .submitter=@ .mem=16000 .gpus=1 .cpus=2
    :: repo=@
    :: src_lang=@
    :: tgt_lang=@
{
    for i in 1 2 3; do
        python $repo/contextual_mt/docmt_cxmi.py \
            --path $checkpoint_dir \
            --source-lang $src_lang --target-lang $tgt_lang \
            --target-context-size $i \
            --source-file $test_src \
            --reference-file $test_tgt \
            --docids-file $test_docids \
                >> $cxmi_results
    done
}

task ProviderTranslate
    < test_src=@GetData
    < test_docids=@GetData
    > test_pred
    :: .submitter=@ .mem=16000 .gpus=0 .cpus=1
    :: repo=@
    :: provider=@
    :: src_lang=@
    :: tgt_lang=@
{
    case $tgt_lang in
        "zh_cn")
            tgt_lang="zh"
        "pt_br")
            tgt_lang="pt"
    esac

    python $repo/benchmark/docmt_provider.py \
        --provider $provider \
        --source-file $test_src \
        --docids-file $test_docids \
        --source-lang $src_lang \
        --target-lang $tgt_lang \
        --translations-file $test_pred
}
    

summary CXMI {
  of MeasureCXMI > cxmi1 cxmi2 cxmi3 {
    head -n 2 $cxmi_results | tail -n 1 | grep -oP 'CXMI: \K[-0-9.]+' > $cxmi1 
    head -n 4 $cxmi_results | tail -n 1 | grep -oP 'CXMI: \K[-0-9.]+' > $cxmi2 
    head -n 6 $cxmi_results | tail -n 1 | grep -oP 'CXMI: \K[-0-9.]+' > $cxmi3
  }

  of ScorePredictions > comet bleu {
      cat $score | grep -oP "COMET = \K[-0-9.]+" > $comet
      cat $score | grep -oP "BLEU = \K[-0-9.]+" > $bleu
  }
}