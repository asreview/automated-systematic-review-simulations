#!/bin/bash

BASE_CFG_FILE=$1

RANGE=("10" "20" "30" "50" "100")

for EPOCH in ${RANGE[*]}; do
    CFG_FILE=${BASE_CFG_FILE%%.ini}_$EPOCH.ini
    cp $BASE_CFG_FILE $CFG_FILE
    echo "epochs=${EPOCH}" >> $CFG_FILE
    ./submit.sh ${CFG_FILE}
done