#!/bin/bash
src=de
tgt=en

export CUDA_VISIBLE_DEVICES=7

data_dir=/opt/data/private/data/deer/dict_trunc/iwslt_data/raw/data-bin
save_dir=/opt/data/private/ckpt/deer/iwslt/deer_src_mask_pruning_reg
xlm_model=/opt/data/private/data/xlmr/xlmr.base/sentencepiece.bpe.model


fairseq-generate $data_dir \
    --user-dir deer_main/src \
    --task deer_task \
    -s $src -t $tgt \
    --gen-subset test \
    --bpe 'sentencepiece' --sentencepiece-model $xlm_model \
    --sacrebleu --remove-bpe 'sentencepiece' \
    --path $save_dir/checkpoint50.pt \
    --iter-decode-max-iter 1 \
    --iter-decode-eos-penalty 0 \
    --beam 1 --remove-bpe \
    --batch-size 64 --quiet \
