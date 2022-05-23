# PACE-Domain-Adaptation

## Instructions

### Step 1: Download Data
Download data into their respective folders in the data folder. Unzip.

### Step 2: Run PACE
We provide two shell scripts: `run_get_results_large.sh` and `run_get_results_small.sh`. These are used to reproduce the large ensemble and small ensemble results in our paper. Run either the `run_get_results_large.sh` or `run_get_results_small.sh`. Do not run both; the small ensemble results are included in the large ensemble results. We tested the code on Linux OS with single P100 or V100 GPU. On a P100 GPU, the inference step takes 9.9 hours for DomainNet and 1.1 hours for Office-Home. DomainNet training takes 1.4 hours per set of results. Office-Home training takes 0.4 hours per set of results. There are two sets of DomainNet results and three sets of Office-Home results. The total time to run `run_get_results_small.sh` is 9.9 + 2 * 1.4 + 1.1 + 3 * 0.4 = 15 hours. `run_get_results_large.sh` takes 4 times as long.

## Description of Files
`data/` This directory contains the text files for the data splits to reproduce our main results. These are identical to other SSDA work.

`get_pre_trained_weights.py` Used to calculate the features for each backbone. The features will be written to the feature_weights folder.

`get_results.py` Trains a linear classifier for SSDA. Prints results. Saves results as a dictionary in *.dic

`get_results_UDA.py` Trains a linear classifier for UDA. Prints results. Saves results as a dictionary in *.dic

`get_results_large.py` Assumes get_results.py has been run for each of the 4 augmentation settings. Reads in the results from *.dic and prints results for the large ensemble. This file takes under a minute to run.

`get_results_large_UDA.py` Assumes get_results_UDA.py has been run for each of the 4 augmentation settings. Reads in the results from *.dic and prints results for the large ensemble. This file takes under a minute to run.
