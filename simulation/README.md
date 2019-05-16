# Simulation - Active Learning with LSTM

This is a simulation in which we explore active-learning solution, using Long short-term memory (LSTM) model.

Active learning is a type of iterative supervised learning method that select the most informative data points to get labeled by an expert during multiple iterations. In scenarios where not enough labled data is available and manually labeling data is expensive, Active learning is expected to show better performance than random approach.

LSTM is a particular type of recurrent neural network with long-term or short-term memory cells. LSTM is perfectly suited for Machine Learning problems in which the prediction of the neural network depends on the historical context of inputs.


We would like to validate our model on several databases with different sizes and various rates of included papers, rather than relying on a single database.


## Base parameters (simulation_LSTM.py)
* Number of simulations : 15 repeats.
* "prior\_included", "prior\_excluded": To compare query strategies, they should have the same starting points. 
Therefore, a list of indices, presenting ten initially included and ten initially excluded papers, are given to the asr software as input.
* "n_queries": The number of active learning iterations is set to 5.
* "n_instances": In each iteration, 100 papers are reviewed.

## Model parameters (sim_settings.ini)

* Model parameters
	* dropout: 0.4
	* lstm\_out\_width: 10

* Fit parameters
	* frac\_included: 0.01
	* batch\_size: 64
	* epochs: 10



## Run on HPC

### Copy files to HPC

You'll need two files: 

* Word embedding file (cc.en.300.vec)
* Data file to simulate (schoot-lgmm-ptsd.csv)

Put these in the directory above the simulation directory. Or adjust simulation_LSTM.py to match the correct path.

### Adjust model parameters

A template with the base parameters is present in sim_settings.ini. The recommended procedure is to:

1. Copy the template to a descriptive filename (e.g. epoch50.ini for simulating 50 epochs).
2. Run ```python simulation_LSTM.py epoch50.ini```
3. Batch files should be located in "batch.slurm_lisa/epoch50".
4. Submit these batch files:

``` bash
for FILE in batch.slurm_lisa/epoch50/batch*.sh; 
    do sbatch $FILE; 
done
```

### Output

1. Output folder should be under epoch50 (or any other name). 
2. Merge runs with `bash ./merge_dir.sh epoch50`
3. Now the combined output should be in output/epoch50, without overwriting previous runs.

# Analysis of the data.

Analysis of the data can be done with the analysis_LSTM.py script:

``` bash
./analysis_LSTM.py dir [dir2] [dir3]...
```