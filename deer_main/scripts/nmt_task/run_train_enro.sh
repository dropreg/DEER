#!/bin/bash
src=en
tgt=ro

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# data_dir=/opt/data/private/data/nmt_data/wmt_enro/data-bin
# data_dir=/opt/data/private/data/nmt_data/wmt16.en-ro.dist/data-bin
# data_dir=/opt/data/private/data/deer/dict_trunc/enro_data/raw/data-bin
data_dir=/opt/data/private/data/deer/dict_trunc/enro_data/kd/data-bin
# save_dir=/opt/data/private/ckpt/deer/enro/deer_src_mask_reg30
save_dir=/opt/data/private/ckpt/deer/enro/deer_kd_pruning_reg30
xlmr_path=/opt/data/private/data/deer/dict_trunc/enro_model/model.pt


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
    --dropout 0.2 --attention-dropout 0.1 \
    --lr 5e-5 \
    --criterion deer_pruning_loss \
    -s $src -t $tgt \
    --max-tokens 6144 \
    --no-progress-bar \
    --seed 64 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --total-num-update 50000 --warmup-updates 500 \
    --max-epoch 100 \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 128 \
    --save-dir $save_dir | tee -a $save_dir/train.log \
