#!/usr/bin/env python


import os
import subprocess
import sys

import shlex

from asreviewcontrib.simulation.batch_entry import _batch_parser
from os.path import isfile


BATCH_TEMPLATE = """\
#!/bin/bash

#SBATCH -t {time}
#SBATCH --tasks-per-node {tasks_per_node}
#SBATCH -N {num_nodes}
#SBATCH -J {job_name}
#SBATCH --output={cur_dir}/hpc_logs/{job_name}.out
#SBATCH --error={cur_dir}/hpc_logs/{job_name}.err

module load eb
# module load Python/3.6.6-intel-2018b

cd {cur_dir}
source asr-env/bin/activate

srun asreview batch {args}

date

"""


def main(cli_args):
    job_name = input("Please enter job name for simulation:  ")
    parser = _batch_parser()
    parser.add_argument("-t", "--time",
                        type=str, default="120:00:00",
                        help="Maximum clock wall time for the optimization"
                        " to run.")
    args = vars(parser.parse_args(cli_args))
    cur_dir = os.getcwd()
    time = args.pop("time")

    log_dir = os.path.join("hpc_batch_files", job_name)
    os.makedirs(log_dir, exist_ok=True)
    batch_file = os.path.join(log_dir, "batch.sh")

    if isfile(batch_file):
        print(f"Error: batch file exists. Delete {batch_file} to continue.")

    if 'model' in args and args['model'].startswith('lstm'):
        tasks_per_node = 4
        num_nodes = 5
    else:
        tasks_per_node = 16
        num_nodes = 1
    batch_str = BATCH_TEMPLATE.format(
        job_name=job_name, tasks_per_node=tasks_per_node,
        num_nodes=num_nodes, args=" ".join(cli_args), cur_dir=cur_dir,
        time=time)

    with open(batch_file, "w") as fp:
        fp.write(batch_str)

    subprocess.run(shlex.split(f"sbatch {batch_file}"))


if __name__ == "__main__":
    main(sys.argv[1:])
