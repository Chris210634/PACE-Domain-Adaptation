#!/bin/bash -l

# DomainNet 1-shot
python get_results.py --augmentation none --dataset multi --shots 1 --save_weights
python get_results.py --augmentation grayscale --dataset multi --shots 1 --save_weights
python get_results.py --augmentation perspective --dataset multi --shots 1 --save_weights
python get_results.py --augmentation randaugment --dataset multi --shots 1 --save_weights

# DomainNet 3-shot
python get_results.py --augmentation none --dataset multi --shots 3 --save_weights
python get_results.py --augmentation grayscale --dataset multi --shots 3 --save_weights
python get_results.py --augmentation perspective --dataset multi --shots 3 --save_weights
python get_results.py --augmentation randaugment --dataset multi --shots 3 --save_weights

# Office-Home 1-shot
python get_results.py --augmentation none --dataset office_home --shots 1 --save_weights
python get_results.py --augmentation grayscale --dataset office_home --shots 1 --save_weights
python get_results.py --augmentation perspective --dataset office_home --shots 1 --save_weights
python get_results.py --augmentation randaugment --dataset office_home --shots 1 --save_weights

# Office-Home 3-shot
python get_results.py --augmentation none --dataset office_home --shots 3 --save_weights
python get_results.py --augmentation grayscale --dataset office_home --shots 3 --save_weights
python get_results.py --augmentation perspective --dataset office_home --shots 3 --save_weights
python get_results.py --augmentation randaugment --dataset office_home --shots 3 --save_weights

# Office-Home 0-shot
python get_results_UDA.py --augmentation none --dataset office_home --save_weights
python get_results_UDA.py --augmentation grayscale --dataset office_home --save_weights
python get_results_UDA.py --augmentation perspective --dataset office_home --save_weights
python get_results_UDA.py --augmentation randaugment --dataset office_home --save_weights

# Large (4x7) Ensemble Results
python get_results_large.py --dataset multi --shots 1
python get_results_large.py --dataset multi --shots 3
python get_results_large.py --dataset office_home --shots 1
python get_results_large.py --dataset office_home --shots 3
python get_results_large_UDA.py --dataset office_home