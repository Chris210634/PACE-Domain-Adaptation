#!/bin/bash -l
module load python3/3.8.10 pytorch/1.9.0
nvidia-smi

sh copy_domainnet_data.sh

python get_pre_trained_weights.py --dataset office_home --augmentation grayscale --domain Real --data_root /scratch/cliao25/office_home --image_list_root data/office_home
python get_pre_trained_weights.py --dataset office_home --augmentation grayscale --domain Clipart --data_root /scratch/cliao25/office_home --image_list_root data/office_home
python get_pre_trained_weights.py --dataset office_home --augmentation grayscale --domain Product --data_root /scratch/cliao25/office_home --image_list_root data/office_home
python get_pre_trained_weights.py --dataset office_home --augmentation grayscale --domain Art --data_root /scratch/cliao25/office_home --image_list_root data/office_home

python get_pre_trained_weights.py --dataset office_home --augmentation randaugment --domain Real --data_root /scratch/cliao25/office_home --image_list_root data/office_home
python get_pre_trained_weights.py --dataset office_home --augmentation randaugment --domain Clipart --data_root /scratch/cliao25/office_home --image_list_root data/office_home
python get_pre_trained_weights.py --dataset office_home --augmentation randaugment --domain Product --data_root /scratch/cliao25/office_home --image_list_root data/office_home
python get_pre_trained_weights.py --dataset office_home --augmentation randaugment --domain Art --data_root /scratch/cliao25/office_home --image_list_root data/office_home

python get_pre_trained_weights.py --dataset office_home --augmentation perspective --domain Real --data_root /scratch/cliao25/office_home --image_list_root data/office_home
python get_pre_trained_weights.py --dataset office_home --augmentation perspective --domain Clipart --data_root /scratch/cliao25/office_home --image_list_root data/office_home
python get_pre_trained_weights.py --dataset office_home --augmentation perspective --domain Product --data_root /scratch/cliao25/office_home --image_list_root data/office_home
python get_pre_trained_weights.py --dataset office_home --augmentation perspective --domain Art --data_root /scratch/cliao25/office_home --image_list_root data/office_home