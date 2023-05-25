#!/bin/bash

if [ $# -ne 1 ]; then
    echo "usage: $0 GENERATE_PY_OUTPUT"
    exit 1
fi

GEN=$1

SYS=$GEN.sys
REF=$GEN.ref
SRC=$GEN.src

if [ $(tail -n 1 $GEN | grep BLEU | wc -l) -ne 1 ]; then
    echo "not done generating"
    exit
fi

grep ^D $GEN | awk -F '\t' '{print $NF}' | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $SYS
grep ^T $GEN | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $REF
grep ^S $GEN | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $SRC
fairseq-score --sys $SYS --ref $REF
