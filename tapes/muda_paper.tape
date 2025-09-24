task ExtractData
    < data_dir=@
    > extract_dir
    :: repo=@
    :: .submitter=@ .mem=8000 .cpus=1
{
    mkdir $extract_dir
    python $repo/scripts/data/ted/extract_ted.py $data_dir $extract_dir
}

task GetData 
    < data_dir=$extract_dir@ExtractData
    > train_src
    > train_tgt
    > train_docids
    > valid_src
    > valid_tgt
    > valid_docids
    > test_src
    > test_tgt
    > test_docids
    > null_file
    :: .submitter=@ .mem=8000 .gres="gpu:1" .cpus=2 .time=0 .exclude=tir-0-19
    :: src_lang=@
    :: tgt_lang=@
    :: repo=@
{
    # copy data files to the ducttape filesystem
    python $repo/scripts/stanza_tokenize.py $data_dir/train.${src_lang} $train_src --detok --lang $src_lang
    python $repo/scripts/stanza_tokenize.py $data_dir/train.${tgt_lang} $train_tgt --detok --lang $tgt_lang
    cp -f $data_dir/train.docids $train_docids
    python $repo/scripts/stanza_tokenize.py $data_dir/dev.${src_lang} $valid_src --detok --lang $src_lang
    python $repo/scripts/stanza_tokenize.py $data_dir/dev.${tgt_lang} $valid_tgt --detok --lang $tgt_lang
    cp -f $data_dir/dev.docids $valid_docids
    python $repo/scripts/stanza_tokenize.py $data_dir/test.${src_lang} $test_src --detok --lang $src_lang
    python $repo/scripts/stanza_tokenize.py $data_dir/test.${tgt_lang} $test_tgt --detok --lang $tgt_lang
    cp -f $data_dir/test.docids $test_docids
    touch $null_file
}

task TrainSentencePiece
    < train_src=@GetData
    < train_tgt=@GetData
    > src_spm 
    > tgt_spm
    > src_vocab
    > tgt_vocab
    :: .submitter=@ .mem=8000 .cpus=1 .time=0 .exclude=tir-0-19
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

task PreparePretrained
    :: .submitter=@ .mem=8000 .cpus=1 .exclude=tir-0-19
    > pretrained_model
    > src_spm
    > tgt_spm 
    > src_vocab
    > tgt_vocab
    :: src_lang=@
    :: tgt_lang=@
{
    if [ "$tgt_lang" == "de" ]; then
        ln -s /projects/tir5/users/patrick/checkpoints/paracrawl/en-de/checkpoint_best.pt $pretrained_model
        ln -s /projects/tir1/corpora/dialogue_mt/paracrawl/en-de/prep/spm.model $src_spm
        ln -s /projects/tir1/corpora/dialogue_mt/paracrawl/en-de/prep/spm.model $tgt_spm
        ln -s /projects/tir1/corpora/dialogue_mt/paracrawl/en-de/prep/dict.en-de.txt $src_vocab
        ln -s /projects/tir1/corpora/dialogue_mt/paracrawl/en-de/prep/dict.en-de.txt $tgt_vocab
    elif [ "$tgt_lang" == "fr" ]; then
        ln -s /projects/tir5/users/patrick/checkpoints/paracrawl/en-fr/checkpoint_best.pt $pretrained_model
        ln -s /projects/tir5/users/patrick/data/paracrawl/en-fr/prep/spm.model $src_spm
        ln -s /projects/tir5/users/patrick/data/paracrawl/en-fr/prep/spm.model $tgt_spm
        ln -s /projects/tir5/users/patrick/data/paracrawl/en-fr/prep/dict.en-fr.txt $src_vocab
        ln -s /projects/tir5/users/patrick/data/paracrawl/en-fr/prep/dict.en-fr.txt $tgt_vocab
    elif [ "$tgt_lang" == "zh_cn" ]; then
        ln -s /projects/tir5/users/patrick/checkpoints/bt_news/pretrain_large_en_zh/checkpoint_best.pt $pretrained_model
        ln -s /projects/tir5/users/patrick/data/bt_news/en-zh/prep/spm.en.model $src_spm
        ln -s /projects/tir5/users/patrick/data/bt_news/en-zh/prep/spm.zh.model $tgt_spm
        ln -s /projects/tir5/users/patrick/data/bt_news/en-zh/prep/dict.en.txt $src_vocab
        ln -s /projects/tir5/users/patrick/data/bt_news/en-zh/prep/dict.zh.txt $tgt_vocab
    elif [ "$tgt_lang" == "ja" ]; then
        ln -s /projects/tir5/users/patrick/checkpoints/jparacrawl/pretrain_large_en_ja/checkpoint_best.pt $pretrained_model
        ln -s /projects/tir5/users/patrick/data/jparacrawl/en-ja/prep/spm.en.model $src_spm
        ln -s /projects/tir5/users/patrick/data/jparacrawl/en-ja/prep/spm.ja.model $tgt_spm
        ln -s /projects/tir5/users/patrick/data/jparacrawl/en-ja/prep/dict.en.txt $src_vocab
        ln -s /projects/tir5/users/patrick/data/jparacrawl/en-ja/prep/dict.ja.txt $tgt_vocab
    elif [ "$tgt_lang" == "pt_br" ]; then
        ln -s /projects/tir5/users/patrick/checkpoints/paracrawl/pretrain_large_en_pt/checkpoint_best.pt $pretrained_model
        ln -s /projects/tir5/users/patrick/data/paracrawl/en-pt/prep/spm.en.model $src_spm
        ln -s /projects/tir5/users/patrick/data/paracrawl/en-pt/prep/spm.pt.model $tgt_spm
        ln -s /projects/tir5/users/patrick/data/paracrawl/en-pt/bin/dict.en.txt $src_vocab
        ln -s /projects/tir5/users/patrick/data/paracrawl/en-pt/bin/dict.pt.txt $tgt_vocab
    else
        echo "unsupported target language for pretrained models"
        exit 1
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
    < src_spm=(
        Pretrain:
            false=$src_spm@TrainSentencePiece
            true=$src_spm@PreparePretrained
        )
    < tgt_spm=(
        Pretrain:
            false=$tgt_spm@TrainSentencePiece
            true=$tgt_spm@PreparePretrained
        )
    > prep_src
    > prep_tgt
    :: .submitter=@ .mem=8000  .cpus=1 .exclude=tir-0-19
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
    < src_spm=(
        Pretrain:
            false=$src_spm@TrainSentencePiece
            true=$src_spm@PreparePretrained
        )
    < tgt_spm=(
        Pretrain:
            false=$tgt_spm@TrainSentencePiece
            true=$tgt_spm@PreparePretrained
        )
    < src_vocab=(
        Pretrain:
            false=$src_vocab@TrainSentencePiece
            true=$src_vocab@PreparePretrained
        )
    < tgt_vocab=(
        Pretrain:
            false=$tgt_vocab@TrainSentencePiece
            true=$tgt_vocab@PreparePretrained
        )
    > bin_dir
    :: .submitter=@ .mem=8000 .cpus=10 .exclude=tir-0-19
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
    < pretrained_model=(
        Pretrain:
            false=$null_file@GetData
            true=$pretrained_model@PreparePretrained
        )
    > checkpoint_dir
    :: .submitter=@ .mem=16000 .gres="gpu:1" .cpus=2 .time=0 .exclude=tir-0-19
    :: src_lang=@
    :: tgt_lang=@
    :: N=@
    :: M=@
    :: multi_encoder=@
    :: pretrained=(
        Pretrain:
            false=false
            true=true
        )
    :: sample_context_size=@
    :: coword_dropout=@
    :: repo=@
    :: seed=@
    
{
    if [ $pretrained = true ]; then 
        lr=1e-4
        arch=contextual_transformer_big
        patience=3
        max_epoch=5 
    else 
        lr=5e-4
        arch=contextual_transformer_iwslt
        patience=5
        max_epoch=75
    fi

    # --eval-bleu \
    # --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    # --eval-bleu-remove-bpe sentencepiece \
    # --eval-bleu-print-samples \
    

    fairseq-train \
        $bin_dir --user-dir $repo/contextual_mt \
        --fp16 \
        --task document_translation \
        --source-context-size $N --target-context-size $M \
        --max-epoch $max_epoch \
        $([ "$sample_context_size" = true ] && echo "--sample-context-size" || echo "") \
        --coword-dropout ${coword_dropout} \
        $([ "$pretrained" = true ] && echo "--finetune-from-model ${pretrained_model}" || echo "") \
        $([ "$multi_encoder" = true ] && echo "--multi-encoder" || echo "") \
        --log-interval 10 \
        --arch $arch --share-decoder-input-output-embed  \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1 \
        --lr $lr --lr-scheduler inverse_sqrt  --warmup-updates 4000 \
        --save-dir $checkpoint_dir --no-epoch-checkpoints \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout 0.3 --weight-decay 0.0001 \
        --max-tokens 4096 --update-freq 8 --patience $patience --seed 42 \
        --seed $seed

    cp $bin_dir/dict.* $bin_dir/spm.* $checkpoint_dir
}

task GeneratePredictions
    < checkpoint_dir=@TrainModel
    < test_src=@GetData
    < test_tgt=@GetData
    < test_docids=@GetData
    > test_pred
    :: .submitter=@ .mem=16000 .gres="gpu:1" .cpus=2 .exclude=tir-0-19
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
        $([[ ! -z "$decode_context_size" ]] && echo "--target-context-size $decode_context_size" || echo "") \
        --reference-file $test_tgt \
        --docids-file $test_docids \
        --source-lang $src_lang --target-lang $tgt_lang \
        --beam 5
}

task MeasureCXMI
    < test_src=@GetData
    < test_tgt=@GetData
    < test_docids=@GetData
    < checkpoint_dir=@TrainModel
    > cxmi_results
    > word_cxmi
    :: .submitter=@ .mem=16000 .gres="gpu:1" .cpus=2
    :: repo=@
    :: src_lang=@
    :: tgt_lang=@
{
    python $repo/contextual_mt/docmt_cxmi.py \
        --path $checkpoint_dir \
        --source-lang $src_lang --target-lang $tgt_lang \
        --source-file $test_src \
        --reference-file $test_tgt \
        --docids-file $test_docids \
        --save-word-level $word_cxmi
            > $cxmi_results
}

task ProviderTranslate
    < test_src=@GetData
    < test_docids=@GetData
    > test_pred
    :: .submitter=@ .mem=16000  .cpus=1
    :: repo=@
    :: provider=@
    :: provider_sentence_level=@
    :: api_key=@
    :: src_lang=@
    :: tgt_lang=@
{
    python $repo/benchmark/docmt_provider.py \
        --provider $provider \
        $([[ ! -z "$api_key" ]] && echo "--api-key $api_key" || echo "") \
        $([ "$provider_sentence_level" = true ] && echo "--context-size 0" || echo "") \
        --source-file $test_src \
        --docids-file $test_docids \
        --source-lang $src_lang \
        --target-lang $tgt_lang \
        --translations-file $test_pred
}

task ScorePredictions
    < test_pred=(
        Provider:
            null=$test_pred@GeneratePredictions
            google=$test_pred@ProviderTranslate
            deepl=$test_pred@ProviderTranslate
    )
    < test_src=@GetData
    < test_tgt=@GetData
    > score
    :: .submitter=@ .mem=16000 .gres="gpu:1" .cpus=2 .exclude=tir-0-19
    :: repo=@
    :: tgt_lang=@
    :: comet_dir=@
{
    python $repo/scripts/score.py $test_pred $test_tgt \
        --src $test_src \
        --comet-dir $comet_dir > $score
}

task TagReference
    < test_src=@GetData
    < test_tgt=@GetData
    < test_docids=@GetData
    < bpe_cxmis=(
        SampleContextSize:
            true=$word_cxmi@MeasureCXMI
            false=/dev/null
        )
    < spm_model=(
        SampleContextSize:
            true=(
                Pretrain:
                    false=$tgt_spm@TrainSentencePiece
                    true=$tgt_spm@PreparePretrained
                )
            false=/dev/null
        )
    > test_src_tok
    > test_tgt_tok
    > tagged_tgt
    > tokenized_cxmis
    :: .submitter=@ .mem=16000 .gres="gpu:1" .time=0 .cpus=4 .exclude=tir-0-19
    :: repo=@
    :: src_lang=@
    :: tgt_lang=@
    :: awesome_cachedir=@
    :: awesome_model="bert-base-multilingual-cased"
{

    python $repo/scripts/stanza_tokenize.py $test_src $test_src_tok --lang $src_lang
    python $repo/scripts/stanza_tokenize.py $test_tgt $test_tgt_tok --lang $tgt_lang

    python $repo/scripts/format_align.py \
        --source-file $test_src_tok \
        --target-file $test_tgt_tok \
        --output formatted_output

    awesome-align \
        --output_file=alignments.out \
        --model_name_or_path=$awesome_model \
        --data_file=formatted_output \
        --extraction ‘softmax’ \
        --batch_size 32  \
        --cache_dir $awesome_cachedir

    python $repo/scripts/tagger.py \
        --src-tok-file $test_src_tok \
        --tgt-tok-file $test_tgt_tok \
        --src-detok-file $test_src \
        --tgt-detok-file $test_tgt \
        --docids-file $test_docids \
        --alignments-file alignments.out \
        --source-lang ${src_lang} --target-lang ${tgt_lang} \
        --source-context-size 1000 \
        --target-context-size 1000 \
        --output $tagged_tgt

    if [ "$bpe_cxmis" ==  "fake_name" ]; then # "/dev/null"
        python $repo/scripts/stanza_convert_cxmi.py \
            --file $test_tgt \
            --tok-file $test_tgt_tok \
            --sp-model $spm_model \
            --cxmis-file $bpe_cxmis \
            --tok-cxmis-file $tokenized_cxmis
    else
        touch $tokenized_cxmis
    fi
}

task TagTranslations 
    < test_docids=@GetData
    < test_src=@GetData
    < test_tgt=@GetData
    < test_src_tok=@TagReference
    < test_pred=(
        Provider:
            null=$test_pred@GeneratePredictions
            google=$test_pred@ProviderTranslate
            deepl=$test_pred@ProviderTranslate
    )
    > test_pred_tok
    > tagged_pred
    :: .submitter=@ .mem=16000 .gres="gpu:1" .cpus=5
    :: repo=@
    :: src_lang=@
    :: tgt_lang=@
    :: awesome_cachedir=@
    :: awesome_model="bert-base-multilingual-cased"
{
    python $repo/scripts/stanza_tokenize.py $test_pred $test_pred_tok --lang $tgt_lang

    python $repo/scripts/format_align.py \
        --source-file $test_src_tok \
        --target-file $test_pred_tok \
        --output formatted_output
    awesome-align \
        --output_file=alignments.out \
        --model_name_or_path=$awesome_model \
        --data_file=formatted_output \
        --extraction ‘softmax’ \
        --batch_size 32  \
        --cache_dir $awesome_cachedir

    python $repo/scripts/tagger.py \
        --src-tok-file $test_src_tok \
        --tgt-tok-file $test_pred_tok \
        --src-detok-file $test_src \
        --tgt-detok-file $test_pred \
        --docids-file $test_docids \
        --alignments-file alignments.out \
        --source-lang ${src_lang} --target-lang ${tgt_lang} \
        --source-context-size 1000 \
        --target-context-size 1000 \
        --output $tagged_pred
}

task CompareMT
    < test_tgt_tok=@TagReference
    < test_pred_tok=@TagTranslations
    < tagged_tgt=@TagReference
    < tagged_pred=@TagTranslations
    > comparemt_output
    :: .submitter=@ .mem=16000  .cpus=1
    :: repo=@
{
    python $repo/scripts/get_label_set.py --tag-file $tagged_tgt --labels-file label.set
    compare-mt $test_tgt_tok $test_pred_tok \
        --output_directory comparemt_dir \
        --compare_word_accuracies bucket_type=multilabel,ref_labels=${tagged_tgt},out_labels="$tagged_pred",label_set=$(cat label.set) \
        > $comparemt_output
}

summary TranslationQuality {
  of ScorePredictions > comet bleu {
      cat $score | grep -oP "COMET = \K[-0-9.]+" > $comet
      cat $score | grep -oP "BLEU = \K[-0-9.]+" > $bleu
  }
}