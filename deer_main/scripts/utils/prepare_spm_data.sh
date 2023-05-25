SRC=de
TGT=en

DATA=/opt/data/private/data/deer/iwslt_data/raw
SPM=/opt/data/private/code/sentencepiece/build/src/spm_encode
MODEL=/opt/data/private/data/xlmr/xlmr.base/sentencepiece.bpe.model

${SPM} --model=${MODEL} < ${DATA}/train.${SRC} > ${DATA}/train.spm.${SRC}
${SPM} --model=${MODEL} < ${DATA}/train.${TGT} > ${DATA}/train.spm.${TGT}
${SPM} --model=${MODEL} < ${DATA}/valid.${SRC} > ${DATA}/valid.spm.${SRC}
${SPM} --model=${MODEL} < ${DATA}/valid.${TGT} > ${DATA}/valid.spm.${TGT}
${SPM} --model=${MODEL} < ${DATA}/test.${SRC} > ${DATA}/test.spm.${SRC}
${SPM} --model=${MODEL} < ${DATA}/test.${TGT} > ${DATA}/test.spm.${TGT}
