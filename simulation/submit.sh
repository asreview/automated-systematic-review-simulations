#!/bin/bash

source cpu-node/bin/activate

CFG_FILES=($*)

if [ $# -lt 1 ]; then
    echo "Supply config file to submit."
    exit 192
fi

DATA=""
LAST_CFG=${CFG_FILES[${#CFG_FILES[@]}-1]}

if [[ ! $LAST_CFG =~ \.ini$ ]]; then
    DATA=$LAST_CFG
    unset CFG_FILES[${#CFG_FILES[@]}-1]
fi

for CFG_FILE in ${CFG_FILES[*]}; do
    if [ ! -f $CFG_FILE ]; then
        echo "Configuration file $CFG_FILE does not exist"
        exit 192
    fi
    ./simulation_LSTM.py $CFG_FILE $DATA
    DIR=batch.slurm_lisa/asr_${CFG_FILE%%.ini}
    for FILE in $DIR/*.sh; do
        sbatch $FILE
    done
done
