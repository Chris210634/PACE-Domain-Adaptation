#!/bin/bash -l
module load python3/3.8.10 pytorch/1.9.0
nvidia-smi

sh copy_domainnet_data.sh

python get_pre_trained_weights.py --dataset multi --augmentation randaugment --domain real --data_root /scratch/cliao25/multi --image_list_root data/multi
python get_pre_trained_weights.py --dataset multi --augmentation randaugment --domain clipart --data_root /scratch/cliao25/multi --image_list_root data/multi
python get_pre_trained_weights.py --dataset multi --augmentation randaugment --domain painting --data_root /scratch/cliao25/multi --image_list_root data/multi
python get_pre_trained_weights.py --dataset multi --augmentation randaugment --domain sketch --data_root /scratch/cliao25/multi --image_list_root data/multi