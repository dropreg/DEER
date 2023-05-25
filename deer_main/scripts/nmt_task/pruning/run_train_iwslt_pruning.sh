#!/bin/bash
src=de
tgt=en

export CUDA_VISIBLE_DEVICES=0,1,2,3

data_dir=/opt/data/private/data/deer/dict_trunc/iwslt_data/raw/data-bin
save_dir=/opt/data/private/ckpt/deer/iwslt/deer_src_mask_pruning_reg
# xlmr_path=/opt/data/private/data/xlmr/xlmr.base/model.pt
xlmr_path=/opt/data/private/data/deer/dict_trunc/iwslt_model/model.pt


fairseq-train $data_dir \
    --restore-file $xlmr_path \
    --reset-optimizer --reset-dataloader --reset-meters \
    --ddp-backend=legacy_ddp \
    --user-dir deer_main/src \
    --task deer_task \
    --arch deer_pruning_transformer \
    --share-all-embeddings \
    --truncate-source \
    --bpe sentencepiece \
    --dropout 0.1 --attention-dropout 0.1 \
    --lr 5e-5 \
    --criterion deer_pruning_loss \
    -s $src -t $tgt \
    --max-tokens 3072 \
    --no-progress-bar \
    --seed 64 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --total-num-update 30000 --warmup-updates 500 \
    --max-epoch 60 \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 128 \
    --save-dir $save_dir | tee -a $save_dir/train.log \
