#!/bin/bash
src=en
tgt=de

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# data_dir=/opt/data/private/data/nmt_data/wmt14_ende/data-bin
# data_dir=/opt/data/private/data/nmt_data/wmt14_ende_distill/data-bin
data_dir=/opt/data/private/data/deer/dict_trunc/wmt14_ende/data-bin
save_dir=/opt/data/private/ckpt/deer/wmt14_ende/deer_src_mask_reg
# xlmr_path=/opt/data/private/data/xlmr/xlmr.base/model.pt
xlmr_path=/opt/data/private/data/deer/dict_trunc/ende_model/model.pt


# --restore-file $xlmr_path \
# --reset-optimizer --reset-dataloader --reset-meters \
fairseq-train $data_dir \
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
    --update-freq 3 \
    --no-progress-bar \
    --seed 64 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --find-unused-parameters \
    --lr-scheduler polynomial_decay --total-num-update 140000 --warmup-updates 1000 \
    --max-epoch 50 \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 128 \
    --save-dir $save_dir | tee -a $save_dir/train.log \
