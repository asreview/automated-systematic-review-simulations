#!/usr/bin/env python

import platform
from os.path import splitext, isfile, join
import sys
import subprocess

from asreview.simulation.batch_generator import batch_from_params

machine_cfg_dir = "machine_cfg"
sim_cfg_dir = "sim_cfg"

args = sys.argv[1:]
n_repeat = 15

# Generate a grid with these variables, different combinations for each run.
var_param = {'simulation': range(n_repeat)}

# Fixed parameters that are the same for each run.
fix_param = {}
if len(args):
    fix_param["config_file"] = args[0]
else:
    fix_param["config_file"] = "sim_settings.ini"

data_path = "../hyperopt/data"

if len(args) < 2:
    data_name = "ptsd"
else:
    data_name = args[1]

data_fp = None
if data_name in ["ptsd", "hall", "ace"]:
    data_file = data_name + ".csv"
elif data_name in ["nagtegaal"]:
    data_file = data_name + ".xlsx"
else:
    data_fp = args[1]

if data_fp is None:
    data_fp = join(data_path, data_file)

if not isfile(data_fp):
    print(f"Data file {data_fp} does not exist.")
    sys.exit(192)

# Define file names and data sources
param_file = "params.csv"
if platform.system() == "Darwin":
    batch_config_file = "parallel.ini"
else:
    batch_config_file = "slurm_lisa.ini"
batch_config_file = join(machine_cfg_dir, batch_config_file)
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
                  param_file, batch_config_file)
