#!/usr/bin/env python


import os
import subprocess
import sys


import shlex


BATCH_TEMPLATE = """\
#!/bin/bash

#SBATCH -t {time}
#SBATCH --tasks-per-node {tasks_per_node}
#SBATCH -N {num_nodes}
#SBATCH -J hyper_{hyper_name}
#SBATCH --output={cur_dir}/hpc_logs/{hyper_name}.out
#SBATCH --error={cur_dir}/hpc_logs/{hyper_name}.err

module load eb
# module load Python/3.6.6-intel-2018b

cd {cur_dir}
source asr-env/bin/activate

srun ./run_hyper.py {args} --n_iter 10000 --mpi

date

"""


def main(cli_args):
    mode = cli_args[0]
    if mode == "hyper-cluster":
        from asreviewcontrib.hyperopt.cluster import _parse_arguments  # noqa
    elif mode == "hyper-active":
        from asreviewcontrib.hyperopt.active_learning import _parse_arguments  # noqa
    elif mode == "hyper-inactive":
        from asreviewcontrib.hyperopt.inactive import _parse_arguments  #noqa
    else:
        print("Error: need one of the following modes: ['hyper-cluster',"
              "'hyper-active', 'hyper-inactive'")
        sys.exit(192)

    parser = _parse_arguments()
    parser.add_argument("-t", "--time",
                        type=str, default="120:00:00",
                        help="Maximum clock wall time for the optimization"
                        " to run.")
    args = vars(parser.parse_args(cli_args[1:]))
    cur_dir = os.getcwd()
    time = args.pop("time")
    hyper_name = "_".join(map(str, args.values()))
    batch_file = os.path.join("hpc_batch_files", hyper_name + ".sh")

    if 'model' in args and args['model'].startswith('lstm'):
        tasks_per_node = 4
        num_nodes = 5
    else:
        tasks_per_node = 16
        num_nodes = 1
    batch_str = BATCH_TEMPLATE.format(
        hyper_name=hyper_name, tasks_per_node=tasks_per_node,
        num_nodes=num_nodes, args=" ".join(cli_args), cur_dir=cur_dir,
        time=time)

    os.makedirs("hpc_logs", exist_ok=True)
    with open(batch_file, "w") as fp:
        fp.write(batch_str)

    subprocess.run(shlex.split(f"sbatch {batch_file}"))


if __name__ == "__main__":
    main(sys.argv[1:])
