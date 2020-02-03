# Automated Systematic Review - Simulation study

This project contains the code of the simulation study for the [Automated
Systematic Review](https://github.com/msdslab/automated-systematic-review)
project. It contains code to run batches of simulation runs in parallel using MPI.

We make use of the SURFSara HPC infrastructure. But with some modifications,
the code can be run on other HPC facilities as well.

## Installation 

The Automated Systematic Review project requires Python 3.6+. To run the code you
also need an implementation of the MPI standard. The most well known standard is OpenMPI.
This is not a python package and should be installed separately.

The simulation project itself can be directly installed with: 

```bash
pip install --user git+https://github.com/msdslab/automated-systematic-review-simulations
```
Dependencies are automatically installed.

## Running a batch

To run a batch of simulations on 4 cores and 12 runs, use the following command:

```bash
mpirun -n 4 asreview batch ${DATA_SET} --log_file ${DIR}/results.log --n_runs 12
```

It will create 12 files in the ${DIR} directory, while running on 4 cores in parallel.

## Related packages

- asreview-visualization
	Package for visualization of log files.

- asreview-hyperopt
	Package for optimizing the parameters for the 
