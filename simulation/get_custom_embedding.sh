#!/bin/bash

if [ $# -lt 3 ]; then
    echo "Error: need three arguments:  [embedding] [Data file in] [Vector file out]"
fi

EMBEDDING=$1
CSV_FILE=$2
CUSTOM_EMBEDDING=$3
FDIR=./fastText-0.2.0
TWORD_FILE=word_file.tmp
TEMBEDDING=embedding.tmp

if [ ! -d $FDIR ]; then
    wget https://github.com/facebookresearch/fastText/archive/v0.2.0.zip
    unzip v0.2.0.zip
    cd $FDIR
    make
    cd -
fi

./token_gen.py "$CSV_FILE" "$TWORD_FILE"
$FDIR/fasttext print-word-vectors "$EMBEDDING" < "$TWORD_FILE" > "$TEMBEDDING"

N_LINES=(`wc -l "$TEMBEDDING"`)
N_LINES=${N_LINES[0]}

{ echo "$N_LINES 300"; cat "$TEMBEDDING"; } > "$CUSTOM_EMBEDDING"

rm $TWORD_FILE
rm $TEMBEDDING
