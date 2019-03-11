#!/bin/bash
#SBATCH -t 02:00:00
#SBATCH --tasks-per-node=15
#SBATCH -J asr_sim_active_lstm

module load eb
    module load Python/3.6.1-intel-2016b

    cd $HOME/asr
    mkdir -p "$TMPDIR"/output
    rm -rf "$TMPDIR"/results.log
    cp -r $HOME/asr/pickle "$TMPDIR"
    cd "$TMPDIR"
    
parallel -j 15 << EOF_PARALLEL
python3 -m asr simulate pickle/ptsd_vandeschoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results60.log &> /dev/null
python3 -m asr simulate pickle/ptsd_vandeschoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results61.log &> /dev/null
python3 -m asr simulate pickle/ptsd_vandeschoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results62.log &> /dev/null
python3 -m asr simulate pickle/ptsd_vandeschoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results63.log &> /dev/null
python3 -m asr simulate pickle/ptsd_vandeschoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results64.log &> /dev/null
python3 -m asr simulate pickle/ptsd_vandeschoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results65.log &> /dev/null
python3 -m asr simulate pickle/ptsd_vandeschoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results66.log &> /dev/null
python3 -m asr simulate pickle/ptsd_vandeschoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results67.log &> /dev/null
python3 -m asr simulate pickle/ptsd_vandeschoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results68.log &> /dev/null
python3 -m asr simulate pickle/ptsd_vandeschoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results69.log &> /dev/null
python3 -m asr simulate pickle/ptsd_vandeschoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results70.log &> /dev/null
python3 -m asr simulate pickle/ptsd_vandeschoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results71.log &> /dev/null
python3 -m asr simulate pickle/ptsd_vandeschoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results72.log &> /dev/null
python3 -m asr simulate pickle/ptsd_vandeschoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results73.log &> /dev/null
python3 -m asr simulate pickle/ptsd_vandeschoot_words_20000.pkl --query_strategy lc  --prior_included 1136 1940 466 4636 2708 4301 1256 1552 4048 3560  --prior_excluded 1989 2276 1006 681 4822 3908 4896 3751 2346 2166  --n_queries 12  --n_instances 40 --log_file output/results74.log &> /dev/null
EOF_PARALLEL

cp -r "$TMPDIR"/output  $HOME/asr

if [ "False" == "True" ]; then
    echo "Job $SLURM_JOBID ended at `date`" | mail $USER -s "Job: asr_sim_active_lstm/4 ($SLURM_JOBID)"
fi
date
