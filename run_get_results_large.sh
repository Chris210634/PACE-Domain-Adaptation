#!/bin/bash -l

# DomainNet Inference
python get_pre_trained_weights.py --dataset multi --augmentation none --domain real --data_root data/multi --image_list_root data/multi
python get_pre_trained_weights.py --dataset multi --augmentation none --domain clipart --data_root data/multi --image_list_root data/multi
python get_pre_trained_weights.py --dataset multi --augmentation none --domain painting --data_root data/multi --image_list_root data/multi
python get_pre_trained_weights.py --dataset multi --augmentation none --domain sketch --data_root data/multi --image_list_root data/multi

python get_pre_trained_weights.py --dataset multi --augmentation grayscale --domain real --data_root data/multi --image_list_root data/multi
python get_pre_trained_weights.py --dataset multi --augmentation grayscale --domain clipart --data_root data/multi --image_list_root data/multi
python get_pre_trained_weights.py --dataset multi --augmentation grayscale --domain painting --data_root data/multi --image_list_root data/multi
python get_pre_trained_weights.py --dataset multi --augmentation grayscale --domain sketch --data_root data/multi --image_list_root data/multi

python get_pre_trained_weights.py --dataset multi --augmentation perspective --domain real --data_root data/multi --image_list_root data/multi
python get_pre_trained_weights.py --dataset multi --augmentation perspective --domain clipart --data_root data/multi --image_list_root data/multi
python get_pre_trained_weights.py --dataset multi --augmentation perspective --domain painting --data_root data/multi --image_list_root data/multi
python get_pre_trained_weights.py --dataset multi --augmentation perspective --domain sketch --data_root data/multi --image_list_root data/multi

python get_pre_trained_weights.py --dataset multi --augmentation randaugment --domain real --data_root data/multi --image_list_root data/multi
python get_pre_trained_weights.py --dataset multi --augmentation randaugment --domain clipart --data_root data/multi --image_list_root data/multi
python get_pre_trained_weights.py --dataset multi --augmentation randaugment --domain painting --data_root data/multi --image_list_root data/multi
python get_pre_trained_weights.py --dataset multi --augmentation randaugment --domain sketch --data_root data/multi --image_list_root data/multi

# Office-Home Inference
python get_pre_trained_weights.py --dataset office_home --augmentation none --domain Real --data_root data/office_home --image_list_root data/office_home
python get_pre_trained_weights.py --dataset office_home --augmentation none --domain Clipart --data_root data/office_home --image_list_root data/office_home
python get_pre_trained_weights.py --dataset office_home --augmentation none --domain Product --data_root data/office_home --image_list_root data/office_home
python get_pre_trained_weights.py --dataset office_home --augmentation none --domain Art --data_root data/office_home --image_list_root data/office_home

python get_pre_trained_weights.py --dataset office_home --augmentation grayscale --domain Real --data_root data/office_home --image_list_root data/office_home
python get_pre_trained_weights.py --dataset office_home --augmentation grayscale --domain Clipart --data_root data/office_home --image_list_root data/office_home
python get_pre_trained_weights.py --dataset office_home --augmentation grayscale --domain Product --data_root data/office_home --image_list_root data/office_home
python get_pre_trained_weights.py --dataset office_home --augmentation grayscale --domain Art --data_root data/office_home --image_list_root data/office_home

python get_pre_trained_weights.py --dataset office_home --augmentation perspective --domain Real --data_root data/office_home --image_list_root data/office_home
python get_pre_trained_weights.py --dataset office_home --augmentation perspective --domain Clipart --data_root data/office_home --image_list_root data/office_home
python get_pre_trained_weights.py --dataset office_home --augmentation perspective --domain Product --data_root data/office_home --image_list_root data/office_home
python get_pre_trained_weights.py --dataset office_home --augmentation perspective --domain Art --data_root data/office_home --image_list_root data/office_home

python get_pre_trained_weights.py --dataset office_home --augmentation randaugment --domain Real --data_root data/office_home --image_list_root data/office_home
python get_pre_trained_weights.py --dataset office_home --augmentation randaugment --domain Clipart --data_root data/office_home --image_list_root data/office_home
python get_pre_trained_weights.py --dataset office_home --augmentation randaugment --domain Product --data_root data/office_home --image_list_root data/office_home
python get_pre_trained_weights.py --dataset office_home --augmentation randaugment --domain Art --data_root data/office_home --image_list_root data/office_home

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