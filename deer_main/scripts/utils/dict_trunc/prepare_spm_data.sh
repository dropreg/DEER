# SRC=de
# TGT=en

# DATA=/opt/data/private/data/deer/dict_trunc/iwslt_data/kd/
# SPM=/opt/data/private/code/sentencepiece/build/src/spm_encode
# MODEL=/opt/data/private/data/xlmr/xlmr.base/sentencepiece.bpe.model

# ${SPM} --model=${MODEL} < ${DATA}/train.${SRC} > ${DATA}/train.spm.${SRC}
# ${SPM} --model=${MODEL} < ${DATA}/train.${TGT} > ${DATA}/train.spm.${TGT}
# ${SPM} --model=${MODEL} < ${DATA}/valid.${SRC} > ${DATA}/valid.spm.${SRC}
# ${SPM} --model=${MODEL} < ${DATA}/valid.${TGT} > ${DATA}/valid.spm.${TGT}
# ${SPM} --model=${MODEL} < ${DATA}/test.${SRC} > ${DATA}/test.spm.${SRC}
# ${SPM} --model=${MODEL} < ${DATA}/test.${TGT} > ${DATA}/test.spm.${TGT}

SRC=en
TGT=de

# DATA=/opt/data/private/data/deer/dict_trunc/wmt14_ende/
DATA=/opt/data/private/data/deer/dict_trunc/wmt14_ende/kd
SPM=/opt/data/private/code/sentencepiece/build/src/spm_encode
MODEL=/opt/data/private/data/xlmr/xlmr.base/sentencepiece.bpe.model

${SPM} --model=${MODEL} < ${DATA}/train.en-de.${SRC} > ${DATA}/train.spm.${SRC}
${SPM} --model=${MODEL} < ${DATA}/train.en-de.${TGT} > ${DATA}/train.spm.${TGT}
${SPM} --model=${MODEL} < ${DATA}/valid.en-de.${SRC} > ${DATA}/valid.spm.${SRC}
${SPM} --model=${MODEL} < ${DATA}/valid.en-de.${TGT} > ${DATA}/valid.spm.${TGT}
${SPM} --model=${MODEL} < ${DATA}/test.en-de.${SRC} > ${DATA}/test.spm.${SRC}
${SPM} --model=${MODEL} < ${DATA}/test.en-de.${TGT} > ${DATA}/test.spm.${TGT}

# SRC=en
# TGT=ro

# DATA=/opt/data/private/data/deer/dict_trunc/enro_data/kd
# SPM=/opt/data/private/code/sentencepiece/build/src/spm_encode
# MODEL=/opt/data/private/data/xlmr/xlmr.base/sentencepiece.bpe.model

# ${SPM} --model=${MODEL} < ${DATA}/train.${SRC}-${TGT}.${SRC} > ${DATA}/train.spm.${SRC}
# ${SPM} --model=${MODEL} < ${DATA}/train.${SRC}-${TGT}.${TGT} > ${DATA}/train.spm.${TGT}
# ${SPM} --model=${MODEL} < ${DATA}/valid.${SRC}-${TGT}.${SRC} > ${DATA}/valid.spm.${SRC}
# ${SPM} --model=${MODEL} < ${DATA}/valid.${SRC}-${TGT}.${TGT} > ${DATA}/valid.spm.${TGT}
# ${SPM} --model=${MODEL} < ${DATA}/test.${SRC}-${TGT}.${SRC} > ${DATA}/test.spm.${SRC}
# ${SPM} --model=${MODEL} < ${DATA}/test.${SRC}-${TGT}.${TGT} > ${DATA}/test.spm.${TGT}
