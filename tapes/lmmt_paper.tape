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
    :: .submitter=shell
    :: src_lang=@
    :: tgt_lang=@
    :: repo=@
{
    # copy data files to the ducttape filesystem
    cp $data_dir/train.${src_lang} $train_src
    cp $data_dir/train.${tgt_lang} $train_tgt
    cp $data_dir/train.docids $train_docids
    cp $data_dir/valid.${src_lang} $valid_src
    cp $data_dir/valid.${tgt_lang} $valid_tgt
    cp $data_dir/valid.docids $valid_docids
    cp $data_dir/test.${src_lang} $test_src
    cp $data_dir/test.${tgt_lang} $test_tgt
    cp $data_dir/test.docids $test_docids
}

task PreparePretrained
    :: .submitter=shell
    > pretrained_model
    > src_spm
    > tgt_spm 
    > src_vocab
    > tgt_vocab
    :: src_lang=@
    :: tgt_lang=@
{
    if [ "$tgt_lang" == "de" ]; then
        echo "FIXME"
        exit 1
    else
        echo "unsupported target language for pretrained models"
        exit 1
    fi 
}

task TrainSentencePiece
    < train_src=@GetData
    < train_tgt=@GetData
    > src_spm 
    > tgt_spm
    > src_vocab
    > tgt_vocab
    :: .submitter=@ .mem=8000 .time=0
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
    :: .submitter=@ .mem=8000 .time=0
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
    :: .submitter=@ .mem=8000 .cpus=10 .time=0
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

# ---- LM Data Section ----

task GetDataMono
    < monodata_dir=@
    > mono_train
    > mono_valid
    :: .submitter=shell
    :: src_lang=@
    :: tgt_lang=@
    :: repo=@
{
    # copy data files to the ducttape filesystem
    cp $monodata_dir/train.${tgt_lang} $mono_train
    cp $monodata_dir/valid.${tgt_lang} $mono_valid
}

task ApplySentencePieceMono
    < mono_train=@GetDataMono
    < mono_valid=@GetDataMono
    < tgt_spm=@TrainSentencePiece
    > prep_mono_train
    > prep_mono_valid
    :: .submitter=@ .mem=8000 .gpus=0 .cpus=1
    :: repo=@
{
    python $repo/scripts/spm_encode.py \
                --model $tgt_spm \
                    < $mono_train \
                    > $prep_mono_train
    python $repo/scripts/spm_encode.py \
                --model $tgt_spm \
                    < $mono_valid \
                    > $prep_mono_valid
}

task BinarizeMono
    < prep_mono_train=@ApplySentencePieceMono
    < prep_mono_valid=@ApplySentencePieceMono
    < tgt_vocab=@TrainSentencePiece
    > bin_dir
    :: .submitter=@ .mem=16000 .cpus=10 .time=0
    :: src_lang=@
    :: tgt_lang=@
    :: repo=@
{
    fairseq-preprocess \
        --only-source \
        --trainpref $prep_mono_train \
        --validpref $prep_mono_valid \
        --srcdict $tgt_vocab \
        --destdir $bin_dir \
        --workers 20 
}

# ---- Trainings -------

task TrainMonoModel
    < bin_dir=@BinarizeMono
    > checkpoint_dir
    :: .submitter=@ .mem=16000 .gres="gpu:1" .cpus=2 .time=0
    :: repo=@
    :: seed=0
{
    fairseq-train $bin_dir \
        --user-dir $repo/contextual_mt \
        --task language_modeling \
        --save-dir $checkpoint_dir \
        --arch transformer_lm_iwslt \
        --share-decoder-input-output-embed \
        --dropout 0.1 \
        --optimizer adam --adam-betas '(0.9, 0.98)' \
        --weight-decay 0.0001 --clip-norm 1. \
        --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
        --tokens-per-sample 512 \
        --sample-break-mode complete_doc \
        --max-tokens 2048 \
        --update-freq 16 \
        --fp16 \
        --max-update 50000
}

task TrainModel
    < bin_dir=@Binarize
    < pretrained_model=(
        Pretrain:
            false=/dev/null
            true=$pretrained_model@PreparePretrained
        )
    < lm_model=(
        TrainLMDecoder:
            false=/dev/null
            true=$checkpoint_dir@TrainMonoModel
        )
    > checkpoint_dir
    :: .submitter=@ .mem=16000 .gres="gpu:1" .cpus=2 .time=0
    :: src_lang=@
    :: tgt_lang=@
    :: N=@
    :: M=@
    :: sample_context_size=@
    :: lm_schedule_type=@
    :: lm_prob=@
    :: pretrained=(
        Pretrain:
            false=false
            true=true
        )
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
        patience=10
        max_epoch=150
        max_tokens=4096
        update_freq=8
    fi

    if [ $lm_model != /dev/null ] && [ $pretrained_model != /dev/null ]; then
        echo "Not supported"
        exit 1
    fi
    if [ $lm_model != /dev/null ]; then
        finetune_model=$lm_model/checkpoint_best.pt
    fi
    if [ $pretrained_model != /dev/null ]; then
        finetune_model=$pretrained_model
        
    fi

    fairseq-train \
        $bin_dir --user-dir $repo/contextual_mt \
        --fp16 \
        --task document_translation \
        --source-context-size $N --target-context-size $M \
        --max-epoch $max_epoch \
        $([ ! -z "$finetune_model" ] && echo "--finetune-from-model ${finetune_model}" || echo "") \
        --lm-schedule-type $lm_schedule_type \
        --lm-prob $lm_prob \
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
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
        --seed $seed
    
    cp $bin_dir/dict.* $bin_dir/spm.* $checkpoint_dir
}

task GeneratePredictions
    < checkpoint_dir=@TrainModel
    < test_src=@GetData
    < test_tgt=@GetData
    < test_docids=@GetData
    > test_pred
    :: .submitter=@ .mem=16000 .gres=gpu:1 .cpus=2 .time=0
    :: src_lang=@
    :: tgt_lang=@
    :: gold_context=@
    :: repo=@
{
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

task MeasurePerplexity
    < test_src=@GetData
    < test_tgt=@GetData
    < test_docids=@GetData
    < checkpoint_dir=@TrainModel
    > perplexity_results
    :: .submitter=@ .mem=16000 .gres="gpu:1" .cpus=2 .time=0
    :: repo=@
    :: src_lang=@
    :: tgt_lang=@
{
    python $repo/contextual_mt/docmt_perplexity.py \
        --path $checkpoint_dir \
        --source-lang $src_lang --target-lang $tgt_lang \
        --source-file $test_src \
        --reference-file $test_tgt \
        --docids-file $test_docids \
            > $perplexity_results
    python $repo/contextual_mt/docmt_perplexity.py \
        --path $checkpoint_dir \
        --source-lang $src_lang --target-lang $tgt_lang \
        --source-file $test_src \
        --reference-file $test_tgt \
        --docids-file $test_docids \
        --no-source \
            >> $perplexity_results
}

task ScorePredictions
    < test_pred=@GeneratePredictions
    < test_src=@GetData
    < test_tgt=@GetData
    > score
    :: .submitter=@ .mem=16000 .gres="gpu:1" .cpus=2 .time=0
    :: repo=@
    :: tgt_lang=@
    :: comet_dir=@
{
    python $repo/scripts/score.py $test_pred $test_tgt \
        --src $test_src \
        --comet-model wmt-large-da-estimator-1719 --comet-path $comet_dir > $score
}


task ContrastiveEvaluation
    < checkpoint_dir=@TrainModel
    > score 
    :: .submitter=@ .mem=16000 .gres="gpu:1" .cpus=2 .time=0
    :: repo=@
    :: contr_base=@
    :: bawden_data=@
    :: src_lang=@
    :: tgt_lang=@
{
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
  < perplexity_results=@MeasurePerplexity[Seed:*]
  > avg_mt_ppl
  > avg_lm_ppl
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
        comet=`cat $file | sed -nr 's/.*COMET = ([-0-9\.]+).*/\1/p'`
        total_bleu=`echo "$total_bleu + $bleu" | bc`
        total_comet=`echo "$total_comet + $comet" | bc`
        i=$((i + 1))
    done
    echo "scale=4; $total_bleu / $i" | bc -l > $avg_bleu
    echo "scale=4; $total_comet / $i" | bc -l > $avg_comet

    i=0
    total_mt_ppl=0
    total_lm_ppl=0
    for file in $perplexity_results
    do
        mt_ppl=`head -n1 $file | sed -nr 's/.*Perplexity: ([0-9\.]+).*/\1/p'`
        lm_ppl=`tail -n1 $file | sed -nr 's/.*Perplexity: ([0-9\.]+).*/\1/p'`
        total_mt_ppl=`echo "$total_mt_ppl + $mt_ppl" | bc`
        total_lm_ppl=`echo "$total_lm_ppl + $lm_ppl" | bc`
        i=$((i + 1))
    done
    echo "scale=4; $total_mt_ppl / $i" | bc -l > $avg_mt_ppl
    echo "scale=4; $total_lm_ppl / $i" | bc -l > $avg_lm_ppl

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
        echo "scale=4; $total_anaph / $i"
        echo "scale=4; $total_lex / $i" 
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
    of AverageResults > BLEU COMET Contr MTPerplexity LMPerplexity {
        cp $avg_bleu $BLEU
        cp $avg_comet $COMET
        cp $avg_contr $Contr
        cp $avg_mt_ppl $MTPerplexity
        cp $avg_lm_ppl $LMPerplexity
    }
}