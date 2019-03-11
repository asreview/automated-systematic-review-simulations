# Sub analysis 1- Active Learning with LSTM

This is a simulation in which we explore active-learning solution, using Long short-term memory (LSTM) model.
Active learning is a type of iterative supervised learning method that select the most informative data points to get labeled by an expert during multiple iterations. In scenarios where not enough labled data is available and manually labeling data is expensive, Active learning is expected to show better performance than random approach.

LSTM is a particular type of recurrent neural network with long-term or short-term memory cells.
LSTM is perfectly suited for Machine Learning problems in which the prediction of the neural network depends on the historical context of inputs.


We would like to validate our model on several databases with different sizes and various rates of included papers, rather than relying on a single database.


## Parameter Grid 
* Query strategy: Least Confidence (LC), and Random
* Number of simulations : Each setting is repeated 50 times.

## Other simulation settings
* "prior_included", "prior_excluded": To compare query strategies, they should have the same starting points. 
Therefore, a list of indices, presenting ten initially included and ten initially excluded papers, are given to the asr software as input.
* "n_queries": The number of active learning iterations is set to 12.
* "n_instances": In each iteration, 40 papers are reviewed.

Above settings are chosen to have a trained dataset of 500 papers within a reasonable number of iterations.

## Run on HPC
### Generate batch files
1. Navigate to "sub_analysis_AL_LSTM" folder.
2. Run ``` python simulation_LSTM.py```
3. The output is located in "batch.slurm_lisa" folder

### Copy files to HPC
In your home directory create a folder including:
``` bash
asr/
├── pickle
|       ├── ptsd_vandeschoot_words_20000.pkl
├── batch.slurm_lisa
```
### Run simulation
1. Navigate to "asr" folder.
2. Run 
``` bash
for FILE in batch.slurm_lisa/asr_sim_lstm/batch*.sh; 
    do sbatch $FILE; 
done
```
3. The output is located in "output" folder
