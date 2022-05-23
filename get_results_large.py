import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import random
import sys
import torch

parser = argparse.ArgumentParser(description='Get Results Large Ensemble')
parser.add_argument('--dataset', type=str, default='multi', help='')
parser.add_argument('--shots', type=int, default=3, help='')
args = parser.parse_args()
print(args)

################################# FUNCTIONS #################################
def make_dataset_fromlist(image_list):
    # print("image_list", image_list)
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list

def get_unlabeled_target_labels(dataset, target, num):
    unlabeled_target_image_list_file_path = 'data/{}/unlabeled_target_images_{}_{}.txt'.format(dataset, target, num)
    _, labels = make_dataset_fromlist(unlabeled_target_image_list_file_path)
    return torch.tensor(labels).long()

############################# MAIN #################################

dataset = args.dataset
shots = args.shots

# (network, feature_size, batch_size, crop_size)
networks = [('convnext_xlarge_384_in22ft1k', 2048, 12, 384),
 ('convnext_xlarge_in22ft1k', 2048, 12, 224),
 ('convnext_xlarge_in22k', 2048, 12, 224),
 ('swin_large_patch4_window7_224', 1536, 12, 224),
 ('swin_large_patch4_window7_224_in22k', 1536, 12, 224),
 ('swin_large_patch4_window12_384', 1536, 12, 384),
 ('swin_large_patch4_window12_384_in22k', 1536, 12, 384)]

if dataset == 'office_home':
    domain_pairs = [('Real','Clipart'),
                   ('Real','Product'),
                   ('Real','Art'),
                   ('Product', 'Real'),
                   ('Product', 'Clipart'),
                   ('Product', 'Art'),
                   ('Art','Product'),
                   ('Art','Clipart'),
                   ('Art','Real'),
                   ('Clipart','Real'),
                   ('Clipart','Art'),
                   ('Clipart','Product')]
    num_classes = 65
    
else:
    assert dataset == 'multi'
    domain_pairs = [('real','clipart'),
                   ('real','painting'),
                   ('painting','clipart'),
                   ('clipart', 'sketch'),
                   ('sketch', 'painting'),
                   ('real', 'sketch'),
                   ('painting','real')]
    num_classes = 126

# Retrieve saved predictions
augmentations = ['none','perspective','randaugment','grayscale']
master_dic = {}
for augmentation in augmentations:
    master_dic[augmentation] = torch.load('{}_{}_pseudo_labeling_results_{}.dic'.format(dataset, shots, augmentation))
    
# Following code asks each 4x7 enasemble members to vote. Majority wins.
print('================================================================================')
print('Ask each 4x7 enasemble members to vote. Majority wins.')
print('================================================================================')
for source, target in domain_pairs:
    print('{}->{}'.format(source[0], target[0]), end = ', ')
print()

for source, target in domain_pairs:
    unlabeled_target_labels = get_unlabeled_target_labels(dataset, target, num=shots)
    
    ################ Find best models by validation acc ######################
    # This step may not be necessary, this only makes a difference if the maority vote is tied. 
    # I want to be consciously making a choice instead of letting argmax choose whichever 
    # model came first in the case of a tie.
    # for source, target in domain_pairs:
    val_accs = []
    for network, _, _, _ in networks:
        val_acc = master_dic['none'][(network, source, target)][-1][2] # [-1] means last result tuple, [2] is the index of validation acc
        val_accs.append(val_acc)
            
    model_id_low_to_high = np.argsort(val_accs) # index of network by low -> high confidence
    ###########################################################################
    
    votes = []
    for net_id in list(model_id_low_to_high)[::-1]:
        # network
        network, inc, bs, cs = networks[net_id]
        for augmentation in augmentations:
            dic = master_dic[augmentation]
            target_acc, source_acc, val_acc, unlabeled_preds, _ = dic[(network, source, target)][-1]
            votes.append(unlabeled_preds)
        
    ### MAJORITY VOTE ###
    S = torch.zeros_like(F.one_hot(votes[0], num_classes=num_classes))
    for vote in votes:
        S = S + F.one_hot(vote, num_classes=num_classes)
    num_votes, prediction = torch.max(S,1)
    assert num_votes.max() == 28

    acc = ((prediction == unlabeled_target_labels).sum()/len(prediction)).item()*100.
    
    print(acc, end = ', ')
print()
print()
    
# Retrieve saved classifier weights
master_dic_D = {}
for augmentation in augmentations:
    master_dic_D[augmentation] = torch.load('{}_{}_pseudo_labeling_classifiers_{}.dic'.format(dataset, shots, augmentation))
    
# Now CONFIDENCE over all 4 x 7 models
print('================================================================================')
print('Average prediction over 4 x 7 models.')
print('================================================================================')
device = 'cpu'
for source, target in domain_pairs:
    print('{}->{}'.format(source[0], target[0]), end = ', ')
print()

for source, target in domain_pairs:
    unlabeled_target_labels = get_unlabeled_target_labels(dataset, target, num=shots)
    y_hat_sum = torch.zeros((unlabeled_target_labels.shape[0], num_classes))
    for augmentation in augmentations:
        for network, inc, bs, cs in networks:
            D, x_tu, x_tv = master_dic_D[augmentation][(network, source, target)]
            D = D.to(device)
            x_tu = x_tu.to(device)
            y_hat = F.softmax(D(x_tu), -1)
            y_hat_sum = y_hat_sum + y_hat.cpu()
    _, unlabeled_preds = torch.max(y_hat_sum, -1)
    prediction = unlabeled_preds.cpu()
    acc = ((prediction == unlabeled_target_labels).sum()/len(prediction)).item()*100.
    print(acc, end = ', ')
print()
print()
