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
    > src_spm
    > tgt_spm
    > src_dict
    > tgt_dict
    > bin_dir
    > pretrained_model
    :: .submitter=@ .mem=8000 .gpus=0 .cpus=1
    :: pretrained=(
        Pretrain:
            false=false
            true=true
        )
    :: pretrained_dir=@
    :: src_lang=@
    :: tgt_lang=@
    :: repo=@
{

    lp_data_dir="${data_dir}/${src_lang}-${tgt_lang}"

    # copy data files to the ducttape filesystem
    ln -s $lp_data_dir/train.${src_lang}-${tgt_lang}.${src_lang} $train_src
    ln -s $lp_data_dir/train.${src_lang}-${tgt_lang}.${tgt_lang} $train_tgt
    ln -s $lp_data_dir/train.${src_lang}-${tgt_lang}.docids $train_docids
    ln -s $lp_data_dir/valid.${src_lang}-${tgt_lang}.${src_lang} $valid_src
    ln -s $lp_data_dir/valid.${src_lang}-${tgt_lang}.${tgt_lang} $valid_tgt
    ln -s $lp_data_dir/valid.${src_lang}-${tgt_lang}.docids $valid_docids
    ln -s $lp_data_dir/test.${src_lang}-${tgt_lang}.${src_lang} $test_src
    ln -s $lp_data_dir/test.${src_lang}-${tgt_lang}.${tgt_lang} $test_tgt
    ln -s $lp_data_dir/test.${src_lang}-${tgt_lang}.docids $test_docids

    if [ $pretrained = true ]; then
        ln -s ${lp_data_dir}/bin-pretrained/ $bin_dir
        ln -s ${lp_data_dir}/prep-pretrained/spm.${src_lang}.model $src_spm
        ln -s ${lp_data_dir}/prep-pretrained/spm.${tgt_lang}.model $tgt_spm
        ln -s ${lp_data_dir}/bin-pretrained/dict.${src_lang}.txt $src_dict
        ln -s ${lp_data_dir}/bin-pretrained/dict.${tgt_lang}.txt $tgt_dict
        ln -s ${pretrained_dir}/${src_lang}-${tgt_lang}/checkpoint_best.pt $pretrained_model
    else
        ln -s ${lp_data_dir}/bin/ $bin_dir
        ln -s ${lp_data_dir}/prep/spm.${src_lang}.model $src_spm
        ln -s ${lp_data_dir}/prep/spm.${tgt_lang}.model $tgt_spm
        ln -s ${lp_data_dir}/bin/dict.${src_lang}.txt $src_dict
        ln -s ${lp_data_dir}/bin/dict.${tgt_lang}.txt $tgt_dict
        touch $pretrained_model
    fi
}


task TrainModel
    < bin_dir=@GetData
    < src_dict=@GetData
    < tgt_dict=@GetData
    < src_spm=@GetData
    < tgt_spm=@GetData
    < pretrained_model=@GetData 
    > checkpoint_dir
    :: .submitter=@ .mem=16000 .gpus=1 .cpus=2
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
        max_epoch=10
        max_tokens=2048
        update_freq=16
    else 
        lr=5e-4
        arch=contextual_transformer_iwslt
        patience=5
        max_epoch=100
        max_tokens=4096
        update_freq=8
    fi

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
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1. \
        --lr $lr --lr-scheduler inverse_sqrt  --warmup-updates 4000 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout 0.3 --weight-decay 0.0001 \
        --max-tokens ${max_tokens} --update-freq ${update_freq} --patience $patience --seed ${seed} \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-remove-bpe sentencepiece \
        --eval-bleu-print-samples \
        --save-dir $checkpoint_dir --no-epoch-checkpoints \
        --seed $seed

    ln -s $src_spm $checkpoint_dir/spm.${src_lang}.model
    ln -s $tgt_spm $checkpoint_dir/spm.${tgt_lang}.model
    ln -s $src_dict $checkpoint_dir/dict.${src_lang}.txt
    ln -s $tgt_dict $checkpoint_dir/dict.${tgt_lang}.txt
}

task GeneratePredictions
    < src_dict=@GetData
    < tgt_dict=@GetData
    < checkpoint_dir=@TrainModel
    < test_src=@GetData
    < test_tgt=@GetData
    < test_docids=@GetData
    > test_pred
    :: .submitter=@ .mem=16000 .gpus=1 .cpus=2
    :: src_lang=@
    :: tgt_lang=@
    :: gold_context=@
    :: repo=@
{
    ln -sf $src_dict $checkpoint_dir/dict.${src_lang}.txt
    ln -sf $tgt_dict $checkpoint_dir/dict.${tgt_lang}.txt
    python $repo/contextual_mt/docmt_translate.py \
        --path $checkpoint_dir \
        --source-file $test_src \
        --predictions-file $test_pred \
        $([ "$gold_context" = true ] && echo "--gold-target-context" || echo "") \
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
    :: .submitter=@ .mem=16000 .gpus=1 .cpus=2
    :: repo=@
    :: src_lang=@
    :: tgt_lang=@
    :: source_decode_size=@
    :: target_decode_size=@
{
    python $repo/contextual_mt/docmt_cxmi.py \
        --path $checkpoint_dir \
        --source-lang $src_lang --target-lang $tgt_lang \
        --source-file $test_src \
        --reference-file $test_tgt \
        --docids-file $test_docids \
        --save-word-level $word_cxmi \
        $([[ ! -z "$source_decode_size" ]] && echo "--source-context-size $source_decode_size" || echo "") \
        $([[ ! -z "$target_decode_size" ]] && echo "--target-context-size $target_decode_size" || echo "") \
            > $cxmi_results
}

task ScorePredictions
    < test_pred=@GeneratePredictions
    < test_src=@GetData
    < test_tgt=@GetData
    > score
    :: .submitter=@ .mem=16000 .gpus=1 .cpus=2
    :: repo=@
    :: tgt_lang=@
    :: comet_dir=@
{
    python $repo/scripts/score.py $test_pred $test_tgt \
        --src $test_src \
        --lang $tgt_lang \
        --comet-model wmt-large-da-estimator-1719 --comet-path $comet_dir > $score
}

task ContrastiveEvaluation
    < src_dict=@GetData
    < tgt_dict=@GetData
    < checkpoint_dir=@TrainModel
    > score 
    :: .submitter=@ .mem=16000 .gpus=1 .cpus=2
    :: repo=@
    :: contr_base=@
    :: bawden_data=@
    :: src_lang=@
    :: tgt_lang=@
{
    ln -sf $src_dict $checkpoint_dir/dict.${src_lang}.txt
    ln -sf $tgt_dict $checkpoint_dir/dict.${tgt_lang}.txt

    if [ "$tgt_lang" == "de" ]; then
        contr_source=$contr_base/contrapro.text.en
        contr_src_ctx=$contr_base/contrapro.context.en
        contr_target=$contr_base/contrapro.text.de
        contr_tgt_ctx=$contr_base/contrapro.context.de
        python $repo/contextual_mt/docmt_contrastive_eval.py \
            --path $checkpoint_dir \
            --source-lang $src_lang --target-lang $tgt_lang \
            --source-file $contr_source \
            --src-context-file $contr_src_ctx \
            --target-file $contr_target \
            --tgt-context-file $contr_tgt_ctx > $score
    else
        python $repo/contextual_mt/docmt_contrastive_eval.py \
            --path $checkpoint_dir \
            --dataset bawden \
            --source-lang ${src_lang} --target-lang ${tgt_lang} \
            --source-file $bawden_data/anaphora.current.en \
            --src-context-file $bawden_data/anaphora.prev.en \
            --target-file $bawden_data/anaphora.current.fr \
            --tgt-context-file $bawden_data/anaphora.prev.fr > $score
        python $repo/contextual_mt/docmt_contrastive_eval.py \
            --path $checkpoint_dir \
            --dataset bawden \
            --source-lang ${src_lang} --target-lang ${tgt_lang} \
            --source-file $bawden_data/lexical_choice.current.en \
            --src-context-file $bawden_data/lexical_choice.prev.en \
            --target-file $bawden_data/lexical_choice.current.fr \
            --tgt-context-file $bawden_data/lexical_choice.prev.fr  >> $score
    fi
}

task AverageResults
  < scores=$score@ScorePredictions[Seed:*]
  < contr_scores=$score@ContrastiveEvaluation[Seed:*]
  > avg_bleu
  > avg_comet
  > avg_contr
  :: .submitter=shell
  :: tgt_lang=@
{
    i=0
    total_bleu=0
    total_comet=0
    for file in $scores
    do
        bleu=`cat $file | sed -nr 's/.*BLEU = ([0-9\.]+).*/\1/p'`
        comet=`cat $file | sed -nr 's/.*COMET = ([0-9\.]+).*/\1/p'`
        total_bleu=`echo "$total_bleu + $bleu" | bc`
        total_comet=`echo "$total_comet + $comet" | bc`
        i=$((i + 1))
    done
    echo "scale=4; $total_bleu / $i" | bc -l > $avg_bleu
    echo "scale=4; $total_comet / $i" | bc -l > $avg_comet

    if [ "$tgt_lang" = fr ]; then
        i=0
        total_anaph=0
        total_lex=0
        for file in $contr_scores
        do
            anaph=`sed -n 1p $file | sed -nr 's/.*Total Acc: ([0-9\.]+).*/\1/p'`
            lex=`sed -n 3p $file | sed -nr 's/.*Total Acc: ([0-9\.]+).*/\1/p'`
            total_anaph=`echo $total_anaph + $anaph | bc`
            total_lex=`echo $total_lex + $lex | bc`
            i=$((i + 1))
        done
        avg_anaph=`echo "scale=4; $total_anaph / $i" | bc -l`
        avg_lex=`echo "scale=4; $total_lex / $i" | bc -l`
        echo "$avg_anaph $avg_lex" > $avg_contr
    else
        i=0
        total_contr=0
        for file in $contr_scores
        do
            contr=`cat $file | sed -nr 's/.*Total Acc: ([0-9\.]+).*/\1/p'`
            total_contr=`echo $total_contr + $contr | bc`
            i=$((i + 1))
        done
        echo "scale=4; $total_contr / $i" | bc -l > $avg_contr
    fi
}

summary Results {
    of AverageResults > BLEU COMET Contr {
        cp $avg_bleu $BLEU
        cp $avg_comet $COMET
        cp $avg_contr $Contr
    }
}

summary CXMI {
    of MeasureCXMI > CXMI {
        echo `cat $cxmi_results | sed -nr 's/.*CXMI: ([0-9\.]+).*/\1/p'` > $CXMI
    }
}