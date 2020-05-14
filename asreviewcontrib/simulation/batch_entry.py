from copy import deepcopy
import logging
import os

import numpy as np
from mpi4py import MPI

from asreview.entry_points.simulate import _simulate_parser
from asreview.entry_points.base import BaseEntryPoint
from asreview.review import review_simulate

from asreviewcontrib.hyperopt.mpi_executor import mpi_worker, mpi_executor


class JobRunner():
    def execute(self, **kwargs):
        review_simulate(**kwargs)


class BatchEntryPoint(BaseEntryPoint):
    description = "Run batches ASReview simulations."

    def __init__(self):
        super(BatchEntryPoint, self).__init__()
        from asreviewcontrib.simulation.__init__ import __version__
        from asreviewcontrib.simulation.__init__ import __extension_name__

        self.extension_name = __extension_name__
        self.version = __version__

    def execute(self, argv):
        parser = _batch_parser()
        args = parser.parse_args(argv)

        args_dict = vars(args)
        server_job = args_dict.pop("server_job")

        verbose = args_dict.get("verbose", 0)
        if verbose == 0:
            logging.getLogger().setLevel(logging.WARNING)
        elif verbose == 1:
            logging.getLogger().setLevel(logging.INFO)
        elif verbose >= 2:
            logging.getLogger().setLevel(logging.DEBUG)

        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            jobs = create_jobs(**args_dict)
            mpi_executor(jobs, JobRunner(), server_job=server_job)
        else:
            mpi_worker(JobRunner())


DESCRIPTION_BATCH = """
Automated Systematic Review (ASReview) batch system for simulation runs.

It has the same interface as the simulation modus, but adds an extra option
(--n_runs) to run a batch of simulation runs with the same configuration.
"""


def _batch_parser():
    parser = _simulate_parser(prog="batch", description=DESCRIPTION_BATCH)
    # Initial data (prior knowledge)
    parser.add_argument(
        "-r", "--n_run",
        default=10,
        type=int,
        help="Number of runs to perform."
    )
    parser.add_argument(
        "--server_job",
        dest='server_job',
        action='store_true',
        help='Run job on the server. It will incur less overhead of used CPUs,'
        ' but more latency of workers waiting for the server to finish its own'
        ' job.'
    )
    return parser


def create_jobs(**kwargs):
    n_runs = kwargs.pop("n_run")
    state_file = kwargs.pop("state_file", None)
    if state_file is None:
        state_file = kwargs.pop("log_file")
    init_seed = kwargs.pop("init_seed", None)
    r = np.random.RandomState(init_seed)

    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    jobs = []
    for i in range(n_runs):
        split_path = os.path.splitext(state_file)
        new_state_file = split_path[0] + f"_{i}" + split_path[1]
        new_kwargs = deepcopy(kwargs)
        if init_seed is not None:
            new_kwargs["init_seed"] = r.randint(0, 99999999)
        new_kwargs["state_file"] = new_state_file
        jobs.append(new_kwargs)
    return jobs
