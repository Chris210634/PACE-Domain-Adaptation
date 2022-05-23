#!/bin/bash -l
module load python3/3.8.10 pytorch/1.9.0
python get_results_UDA.py --augmentation none --dataset office_home --save_weights