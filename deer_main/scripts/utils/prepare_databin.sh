SRC=de
TGT=en

DATA=/opt/data/private/data/deer/iwslt_data/raw/
MODEL=/opt/data/private/data/xlmr/xlmr.base/

fairseq-preprocess \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --trainpref ${DATA}/train.spm \
  --validpref ${DATA}/valid.spm \
  --testpref ${DATA}/test.spm \
  --destdir ${DATA}/data-bin \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict ${MODEL}/dict.txt \
  --tgtdict ${MODEL}/dict.txt \
  --workers 40 \
