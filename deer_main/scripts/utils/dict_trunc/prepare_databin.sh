# SRC=de
# TGT=en

# SRC=en
# TGT=ro

SRC=en
TGT=de

# DATA=/opt/data/private/data/deer/dict_trunc/iwslt_data/kd/
# DATA=/opt/data/private/data/deer/dict_trunc/wmt14_ende/
# DATA=/opt/data/private/data/deer/dict_trunc/enro_data/raw
# DATA=/opt/data/private/data/deer/dict_trunc/enro_data/kd
DATA=/opt/data/private/data/deer/dict_trunc/wmt14_ende/kd


python deer_main/src/preprocess.py \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --trainpref ${DATA}/train.spm \
  --validpref ${DATA}/valid.spm \
  --testpref ${DATA}/test.spm \
  --destdir ${DATA}/data-bin \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict ${DATA}/dict.txt \
  --tgtdict ${DATA}/dict.txt \
  --workers 40 \
