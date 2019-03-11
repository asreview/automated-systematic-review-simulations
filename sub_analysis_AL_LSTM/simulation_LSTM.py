import sys
import os
#sys.path.append(os.path.abspath(os.curdir))
sys.path.append(os.path.abspath(os.pardir))
from hpc.batch_generator import *

params = {'simulation': range(50),
          'query_strategy': ['random', 'lc']
         }
fix_args ={}
fix_args["prior_included"] = '1136 1940 466 4636 2708 4301 1256 1552 4048 3560'
fix_args["prior_excluded"] = '1989 2276 1006 681 4822 3908 4896 3751 2346 2166'
fix_args["n_queries"] = 12
fix_args["n_instances"] = 40

# sub_analysis_LSTM/
params_file_path = "params.csv"
data_file_path = "pickle/ptsd_vandeschoot_words_20000.pkl"
config_file_path = "slurm_lisa.ini"



generate_params(params, fix_args, params_file_path)
generate_shell_script(data_file_path, params_file_path, config_file_path)