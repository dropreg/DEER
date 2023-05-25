#!/bin/bash
src=src
tgt=tgt

export CUDA_VISIBLE_DEVICES=7


data_dir=/opt/data/private/data/glge/glge-released-dataset/easy/msnews_data/bart_distill_data/processed/data_bin
save_dir=/opt/data/private/ckpt/deer/glge/msnews_distill_pruning

fairseq-generate $data_dir \
    --user-dir deer_main/src \
    --task deer_glge_task \
    -s $src -t $tgt \
    --gen-subset test \
    --bpe gpt2 \
    --truncate-source \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --gpt2-encoder-json deer_main/scripts/glge_task/encoder.json \
    --gpt2-vocab-bpe deer_main/scripts/glge_task/vocab.bpe \
    --path $save_dir/checkpoint40.pt \
    --iter-decode-max-iter 3 \
    --iter-decode-force-max-iter \
    --iter-decode-eos-penalty 0 \
    --beam 1 --remove-bpe \
    --skip-invalid-size-inputs-valid-test \
    --batch-size 64 > $save_dir/inf.log

grep ^S $save_dir/inf.log | cut -f2 > $save_dir/test.src 
grep ^T $save_dir/inf.log | cut -f2 > $save_dir/test.tgt 
grep ^D $save_dir/inf.log | cut -f3 > $save_dir/test.hypo

python deer_main/scripts/glge_task/poster_process.py $save_dir/test.hypo  $save_dir/test.hypo.de

files2rouge $save_dir/test.tgt $save_dir/test.hypo.de --ignore_empty_summary

