#!/bin/bash
src=src
tgt=tgt

export CUDA_VISIBLE_DEVICES=7

data_dir=/opt/data/private/data/glge/glge-released-dataset/easy/msqg_data/bart_distill_data/processed/data_bin
save_dir=/opt/data/private/ckpt/deer/glge/msqg_distill_pruning


fairseq-generate $data_dir \
    --user-dir deer_main/src \
    --task deer_glge_task \
    -s $src -t $tgt \
    --gen-subset test \
    --bpe gpt2 \
    --truncate-source \
    --gpt2-encoder-json deer_main/scripts/glge_task/encoder.json \
    --gpt2-vocab-bpe deer_main/scripts/glge_task/vocab.bpe \
    --path $save_dir/checkpoint50.pt \
    --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 \
    --beam 1 --remove-bpe \
    --skip-invalid-size-inputs-valid-test \
    --batch-size 64 > $save_dir/inf.log

grep ^S $save_dir/inf.log | cut -f2 > $save_dir/test.src 
grep ^T $save_dir/inf.log | cut -f2 > $save_dir/test.tgt 
grep ^D $save_dir/inf.log | cut -f3 > $save_dir/test.hypo

python deer_main/scripts/glge_task/poster_process.py $save_dir/test.hypo  $save_dir/test.hypo.de

/opt/conda/envs/py2/bin/python /opt/data/private/code/ctc_new/fairseq/examples/huawei_pretrain/scripts/glge/eval/qg/eval.py \
--out $save_dir/test.hypo.de \
--src $save_dir/test.src \
--tgt $save_dir/test.tgt

nlg-eval --hypothesis=$save_dir/test.hypo.de --references=$save_dir/test.tgt \
