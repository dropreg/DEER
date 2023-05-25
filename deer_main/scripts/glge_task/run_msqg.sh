#!/bin/bash
src=src
tgt=tgt

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# data_dir=/opt/data/private/data/glge/glge-released-dataset/easy/msqg_data/train_data/processed/data_bin
data_dir=/opt/data/private/data/glge/glge-released-dataset/easy/msqg_data/bart_distill_data/processed/data_bin
save_dir=/opt/data/private/ckpt/deer/glge/msqg_distill_pruning
roberta_path=/opt/data/private/data/roberta.base/model.pt


fairseq-train $data_dir \
    --restore-file $roberta_path \
    --reset-optimizer --reset-dataloader --reset-meters \
    --ddp-backend=legacy_ddp \
    --user-dir deer_main/src \
    --task deer_glge_task \
    --arch deer_pruning_transformer \
    --share-all-embeddings \
    --truncate-source \
    --bpe gpt2 \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --dropout 0.1 --attention-dropout 0.1 \
    --lr 5e-5 \
    --criterion deer_pruning_loss \
    -s $src -t $tgt \
    --max-tokens 4096 \
    --no-progress-bar \
    --seed 64 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --total-num-update 25000 --warmup-updates 500 \
    --max-epoch 50 \
    --skip-invalid-size-inputs-valid-test \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 128 \
    --save-dir $save_dir | tee -a $save_dir/train.log \
