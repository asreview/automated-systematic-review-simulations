#!/bin/bash

function get_res_id {
    FILE=$1
    ID=${FILE%.log}
    ID=${ID##*results}
    echo $ID
}

function get_highest_id {
    MAX_ID=0
    FILES=$*
    for FILE in ${FILES}; do
        NEW_ID=`get_res_id $FILE`
        if [ $NEW_ID -gt $MAX_ID ]; then
            MAX_ID=$NEW_ID
        fi
    done
    echo $MAX_ID
}

N_DIRS=$#
DIRS=($*)

BASE_DST_DIR="./output"


for DIR in ${DIRS[*]}; do
    SRC_RES_FILES=($DIR/results*.log)
    ALL_SRC_FILES=($DIR/*)
    DST_DIR="$BASE_DST_DIR/`basename $DIR`"
    DST_RES_FILES=($DST_DIR/results*.log)
    ALL_DST_FILES=($DST_DIR/*)
    
    if [ ! -d $DIR ]; then
        echo "Warning: directory $DIR doesn't exist"
    elif [ "${SRC_RES_FILES[*]}" != "${ALL_SRC_FILES[*]}" ]; then
        echo "Warning: directory $DIR contains more than results or no results at all"
    elif [ ! -f $SRC_RES_FILES ]; then
        echo "Warning: directory $DIR doesn't have results, skipping"
    elif [ ! -d $DST_DIR ]; then
        echo "For directory `basename $DIR`, destination directory doesn't exist, copying..."
        mkdir -p $DST_DIR
        cp ${SRC_RES_FILES[*]} $DST_DIR
        rm -r $DIR
    elif [ "${DST_RES_FILES[*]}" != "${ALL_DST_FILES[*]}" -a "${ALL_DST_FILES[*]}" != "$DST_DIR/*" ]; then
        echo "Warning: destination directory ${DST_DIR} has already data in it that are not results."
    elif [ ! -f $DST_RES_FILES ]; then
        echo "Directory $DST_DIR doesn't have anything in it, copying..."
        cp ${SRC_RES_FILES[*]} $DST_DIR
        rm -r $DIR
    else
        CUR_ID=`get_highest_id ${DST_RES_FILES[*]}`
        let "CUR_ID++"
        echo "Starting at id $CUR_ID"
        for FILE in ${SRC_RES_FILES[*]}; do
            cp $FILE $DST_DIR/results${CUR_ID}.log
            let "CUR_ID++"
        done
        rm -r $DIR
    fi
done
