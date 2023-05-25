#!/bin/bash
src=en
tgt=ro

export CUDA_VISIBLE_DEVICES=7
export CUDA_LAUNCH_BLOCKING=1

data_dir=/opt/data/private/data/deer/dict_trunc/enro_data/raw/data-bin
# save_dir=/opt/data/private/ckpt/deer/enro/deer_src_mask
save_dir=/opt/data/private/ckpt/deer/enro/deer_src_mask_reg30
# save_dir=/opt/data/private/ckpt/deer/enro/deer_kd_pruning_reg30


fairseq-generate $data_dir \
    --user-dir deer_main/src \
    --task deer_task \
    -s $src -t $tgt \
    --gen-subset test \
    --bpe 'sentencepiece' --sentencepiece-model /opt/data/private/data/xlmr/xlmr.base/sentencepiece.bpe.model \
    --sacrebleu --remove-bpe 'sentencepiece' \
    --path $save_dir/checkpoint80.pt \
    --iter-decode-max-iter 3 \
    --iter-decode-eos-penalty 0 \
    --beam 1 --remove-bpe \
    --batch-size 64 > $save_dir/gen.txt

bash deer_main/scripts/compound_split_bleu_detoken.sh $save_dir/gen.txt
