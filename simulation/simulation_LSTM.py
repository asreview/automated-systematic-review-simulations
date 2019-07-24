#!/usr/bin/env python

import sys
import subprocess

from pargrid import batch_from_params
from os.path import splitext, isfile, join

args = sys.argv[1:]
n_repeat = 15

# Generate a grid with these variables, different combinations for each run.
var_param = {'simulation': range(n_repeat)}

# Fixed parameters that are the same for each run.
fix_param = {}
fix_param["query_strategy"] = "rand_max"
fix_param["n_prior_included"] = 10
fix_param["n_prior_excluded"] = 10
fix_param["n_queries"] = 15
fix_param["n_instances"] = 100
if len(args):
    fix_param["config_file"] = args[0]
else:
    fix_param["config_file"] = "sim_settings.ini"

data_path = "../data"

data_file = join("ptsd", "schoot-lgmm-ptsd.csv")
data_fp = None
if len(args) >= 2:
    data_name = args[1]
    if data_name == "ptsd":
        data_file = join("ptsd", "schoot-lgmm-ptsd.csv")
    elif data_name == "ptsd_new":
        data_file = join("ptsd", "raoulduplicates.csv")
    elif data_name == "statins":
        data_file = join("cohen", "Statins.csv")
    elif data_name == "ace":
        data_file = join("cohen", "ACEInhibitors.csv")
    elif data_name == "depression":
        data_file = join("depression", "Depression Cuipers et al. 2018.csv")
    else:
        data_fp = args[1]

if data_fp is None:
    data_fp = join(data_path, data_file)

if not isfile(data_fp):
    print(f"Data file {data_fp} does not exist.")
    sys.exit(192)

# Define file names and data sources
param_file = "params.csv"
batch_config_file = "slurm_lisa.ini"
base_embedding_file = "../../cc.en.300.bin"

embedding_file = splitext(data_fp)[0]+".vec"
if not isfile(embedding_file):
    command = ["./get_custom_embedding.sh"]
    command.append(data_fp)
    command.append(base_embedding_file)
    command.append(embedding_file)
    subprocess.run(command)

# Create a batch from the above file names and data sources.
batch_from_params(var_param, fix_param,
                  data_fp, embedding_file,
                  param_file, batch_config_file,
                  use_pickle=False,
                  )
