#!/bin/bash -l
module load python3/3.8.10 pytorch/1.9.0
python get_results.py --augmentation randaugment --dataset office_home --shots 1 --save_weights