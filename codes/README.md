## Codes Overview

This repository consists of 3 parts: part 1 is the code for simulation, part 2 is that for experiment, and part 3 is for models. The `./utils.py` file contains useful functions for simulation and experiment.

### PART 1. Simulation

1. `./simulation/main.ipynb`: the main file of the simulation studies
2. `./simulation/generate_data.py`: the file for synthetic data generation
3. `./simulation/plot_RMSE_results.ipynb`: the file to summarize the 1000 results into Table 1 of the paper
4. `./simulation/plot_MAC_results.ipynb`: the file to summarize the 1000 results into Table 2 of the paper
5. `./simulation/results/*.csv`: the simulation results for 1000 replications


### PART 2. Experiment


1. `./experiment/main.ipynb`: the main file of the experiment studies
2. `./experiment/transfer_learning.ipynb`: the transfer learning of the experiment studies to train deep learning models serving as crowd annotators
3. `./experiment/expe_utils.py`: the useful functions for the experiment studies
4. `./experiment/feat_X.npy`: the feature vector of the unlabeled dataset
5. `./experiment/crowd labels/*.npy`: the crowd labels provided by the corresponding crowd annotator (deep learning model)


### PART 3. Model

1. `./model/BaseModel.py`: the base model for the following initial/one-step/two-step/multiple-step model
2. `./model/Initial.py`: provide initial estimators $\widetilde \beta$ and $\widetilde \sigma$
3. `./model/OS.py`: provide one-step (OS) estimators $\widehat \beta^\text{OS}$ and $\widehat \sigma^\text{OS}$
4. `./model/MS.py`: provide multiple-step (MS) estimators $\widehat \beta^\text{MS}_{(s)}$ and $\widehat \sigma^\text{TS}_{(s)}$. When $s=2$, the MS estimators are TS estimators.
5. `./model/ORACLE_beta.py`: provide the ID estimator $\widehat \beta^\text{ID}$
6. `./model/ORACLE_sigma.py`: provide the ID estimator $\widehat \sigma^\text{ID}$

