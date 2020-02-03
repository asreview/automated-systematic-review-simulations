from copy import deepcopy
import logging
import os

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
            mpi_executor(jobs, JobRunner(), server_job=True)
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
        "-r", "--n_runs",
        default=10,
        type=int,
        help="Number of runs to perform."
    )
    return parser


def create_jobs(**kwargs):
    n_runs = kwargs.pop("n_runs")
    log_file = kwargs.pop("log_file")
    jobs = []
    for i in range(n_runs):
        split_path = os.path.splitext(log_file)
        new_log_file = split_path[0] + f"_{i}" + split_path[1]
        new_kwargs = deepcopy(kwargs)
        new_kwargs["log_file"] = new_log_file
        jobs.append(new_kwargs)
    return jobs
