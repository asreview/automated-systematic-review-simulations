# python -m asr simulate pickle/schoot-lgmm-ptsd_words_20000.pkl --query_strategy max_sampling  --prior_included 1136 1940  466 4636 2708 4301 1256 1552 4048 3560 2845 2477 4145 1960  --prior_excluded 2820 2035 2823 2318 1153 2973 3455  477 2043 1727 2224 3173 4173 3051  --n_queries 2  --n_instances 100  --config_file sim_settings.ini --log_file test/results0.log

python -m asr simulate pickle/schoot-lgmm-ptsd_words_20000.pkl --n_prior_included 10  --n_prior_excluded 10  --n_queries 2  --n_instances 50  --config_file sim_settings.ini --log_file test/results0.log

# python -m asr simulate pickle/Statins_words_20000_+0.pkl  --n_prior_included 10  --n_prior_excluded 10  --n_queries 3  --n_instances 50  --config_file sim_settings.ini --log_file test/results0.log

# python -m asr simulate pickle/ACEInhibitors_words_20000_+50.pkl  --n_prior_included 10  --n_prior_excluded 10  --n_queries 3  --n_instances 50  --config_file sim_settings.ini --log_file test/results0.log


# python -m asr simulate 'pickle/Depression Cuipers et al. 2018_words_20000.pkl'  --n_prior_included 10  --n_prior_excluded "10"  --n_queries 3  --n_instances 50  --config_file sim_settings.ini --log_file test_depression_l1_half/results0.log

# python -m asr simulate 'pickle/schoot-lgmm-ptsd_words_20000.pkl'  --model nb --n_prior_included 10  --n_prior_excluded "10"  --n_queries 10  --n_instances 50 --log_file test/results0.log -q max_sampling



# 1136 1940  466 4636 2708 4301 1256 1552 4048 3560 2845 2477 4145 1960
# 2820 2035 2823 2318 1153 2973 3455  477 2043 1727 2224 3173 4173 3051
