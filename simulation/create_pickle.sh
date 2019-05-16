#!/bin/bash

FILES=(../data/*/*.csv)
COMMAND_FILE=commands.sh
PRE_FILE=pre_pickle.sh
CFG_FILE=lisa_pickle.ini
WD=`pwd`

cat > $PRE_FILE << EOF_CAT
cd $WD
EOF_CAT

cat > $CFG_FILE << EOF_CAT
[BACKEND]

backend = parallel

[BATCH_OPTIONS]

num_cores_simul = 1

num_tasks_per_node = 10
job_name = pickle
clock_wall_time = 1:00:00

EOF_CAT

touch $COMMAND_FILE

echo ${FILES[*]}
for FILE in "${FILES[@]}"; do
    VEC_FILE=${FILE%%.csv}.vec
    if [ ! -f "$VEC_FILE" ]; then
        echo "$VEC_FILE"
        continue
    fi
    echo "pickle_asr '$FILE' '$VEC_FILE'" >> $COMMAND_FILE
done

batchgen -f $COMMAND_FILE $CFG_FILE -pre $PRE_FILE

rm $COMMAND_FILE $PRE_FILE $CFG_FILE
