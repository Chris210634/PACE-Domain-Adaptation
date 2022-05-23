#!/bin/bash -l
module load python3/3.8.10 pytorch/1.9.0
python get_results.py --augmentation perspective --dataset office_home --shots 1 --save_weights