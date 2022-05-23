#!/bin/bash -l

# DomainNet
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

# Office-Home
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