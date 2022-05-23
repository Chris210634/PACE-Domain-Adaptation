#!/bin/bash -l
module load python3/3.8.10 pytorch/1.9.0
python scramble.py
python get_results.py --augmentation none --dataset multi --shots 3
python scramble.py
python get_results.py --augmentation none --dataset multi --shots 3
python scramble.py
python get_results.py --augmentation none --dataset multi --shots 3