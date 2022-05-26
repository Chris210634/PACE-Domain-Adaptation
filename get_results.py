from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import random
import sys
import PIL
from PIL import Image
import json
import torch
import torchvision
import torchvision.transforms as T
from timm import create_model
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

parser = argparse.ArgumentParser(description='Get Results Small Ensemble')
parser.add_argument('--augmentation', type=str, default='none', help='')
parser.add_argument('--dataset', type=str, default='multi', help='')
parser.add_argument('--shots', type=int, default=3, help='')

parser.add_argument('--save_weights', action='store_true', default=False,
                    help='save classifier weights or not')
parser.add_argument('--PADD', action='store_true', default=False,
                    help='Project away domain direction(PADD) instead of using CORAL')

# Hyperparameters
parser.add_argument('--T', type=int, default=30, help='')
parser.add_argument('--eta0', type=float, default=40, help='')
parser.add_argument('--eta1T', type=float, default=80, help='')
parser.add_argument('--alpha0', type=float, default=0.4, help='')
parser.add_argument('--beta0', type=float, default=0.2, help='')
parser.add_argument('--alpha1T', type=float, default=0.1, help='')
parser.add_argument('--beta1T', type=float, default=0.05, help='')
parser.add_argument('--gamma1T', type=float, default=0.9, help='')
parser.add_argument('--taus1T', type=float, default=0.8, help='')
parser.add_argument('--tautu110', type=float, default=0.9, help='')
parser.add_argument('--tautu1120', type=float, default=0.8, help='')
parser.add_argument('--tautu2130', type=float, default=0.7, help='')
args = parser.parse_args()
print(args)

#################################
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
#################################

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

def coral(source_features, val_target_features, unlabeled_target_features, labeled_target_features):
    x_s = torch.cat((source_features.to(device), labeled_target_features.to(device)))
    x_t = unlabeled_target_features.to(device)
    x_tv = val_target_features.to(device)
    x_s_n = x_s - x_s.mean(0) # centered source
    x_t_n = x_t - x_t.mean(0) # centered target
    x_tv = x_tv - x_t.mean(0)

    x_s_cov = torch.matmul(x_s_n.T, x_s_n) / (x_s_n.shape[0] - 1.)
    x_s_cov = x_s_cov + 0.01 * torch.eye(x_s_cov.shape[0]).to(device)
    x_t_cov = torch.matmul(x_t_n.T, x_t_n) / (x_t_n.shape[0] - 1.)
    x_t_cov = x_t_cov + 0.01 * torch.eye(x_t_cov.shape[0]).to(device)

    x_s_cov_sqrt = torch.tensor(sqrtm(x_s_cov.cpu())).to(device)
    x_s_cov_sqrt_inv = x_s_cov_sqrt.inverse()
    x_s_whitened = torch.matmul(x_s_n, x_s_cov_sqrt_inv.float()) # whiten
    x_t_cov_sqrt = torch.tensor(sqrtm(x_t_cov.cpu())).to(device)
    x_s = torch.matmul(x_s_whitened, x_t_cov_sqrt.float()) # recolor with target variance

    x_tu = x_t_n # centered target
    x_t = x_s[source_features.shape[0]:] # target
    x_s = x_s[:source_features.shape[0]] # source
    
    # target unlabeled, target labeled, source, target validation
    return x_tu, x_t, x_s, x_tv

def padd(source_features, val_target_features, unlabeled_target_features, labeled_target_features, iters=30):
    def project_onto_vector(a, b):
        bu = F.normalize(b)  # project onto b
        a_n = F.normalize(a) 
        cos_theta = torch.matmul(a_n, bu.T) # cos theta
        a_norm = a.square().sum(-1).sqrt() # norm
        return torch.unsqueeze(cos_theta.view(-1) * a_norm, 0).T * bu
    x_s = source_features.to(device)
    x_tv = val_target_features.to(device)
    x_tu = unlabeled_target_features.to(device)
    x_t = labeled_target_features.to(device)
    x_labeled = torch.cat((x_s, x_t))
    
    lam_l1=0.0002
    for _ in range(iters):
        # Train binary linear discriminator between source and target domains
        D = nn.Linear(inc, 1, bias=False).to(device)
        D.to(device)
        optimizer_d = optim.SGD(D.parameters(), lr=4.0, momentum=0.9, weight_decay=0.000, nesterov=True)
        for _ in range(200):
            y_hat_s = torch.sigmoid(D(x_labeled))  # classify as zero
            y_hat_tu = torch.sigmoid(D(x_tu))# classify as one

            optimizer_d.zero_grad()
            loss = 0.5*F.binary_cross_entropy(y_hat_s.view(-1), torch.zeros_like(y_hat_s.view(-1)))
            loss += 0.5*F.binary_cross_entropy(y_hat_tu.view(-1), torch.ones_like(y_hat_tu.view(-1)))
            loss += lam_l1*torch.abs(D.weight.view(-1)).sum()
            loss.backward()
            optimizer_d.step()

        x_labeled = F.normalize(x_labeled - project_onto_vector(x_labeled, D.weight.detach()))
        x_tu = F.normalize(x_tu - project_onto_vector(x_tu, D.weight.detach()))
        x_tv = F.normalize(x_tv - project_onto_vector(x_tv, D.weight.detach()))

    x_t = x_labeled[source_features.shape[0]:] # target
    x_s = x_labeled[:source_features.shape[0]] # source
    # target unlabeled, target labeled, source, target validation
    return x_tu, x_t, x_s, x_tv

def train_classifer(inc, x_s, x_t, y_s, y_t, iters=400, alpha=0.4, beta=0.2, lr=40.0, device='cuda'):
    '''x_s, x_t already normalized'''
    # global num_classes
    D = nn.Linear(inc, num_classes, bias=False).to(device)
    D.to(device)
    optimizer_d = optim.SGD(D.parameters(), lr=lr, momentum=0.9, weight_decay=0.000, nesterov=True)

    for _ in range(iters):
        y_hat_s = D(x_s)
        y_hat_t = D(x_t)

        optimizer_d.zero_grad()
        loss = alpha*F.cross_entropy(y_hat_s, y_s)
        loss += beta*F.cross_entropy(y_hat_t, y_t)
        loss.backward()
        optimizer_d.step()
    return D

def fixmatch(D, x_s, x_t, x_tv, x_tu, y_s, y_t,
             iters=200, alpha=0.1, beta=0.05, gamma=0.9, lr=80.0,
            fixmatch_iters=15, source_thresh=0.8, target_thresh=0.8, device='cuda'):
    D.to(device)
    optimizer_d = optim.SGD(D.parameters(), lr=lr, momentum=0.9, weight_decay=0.000, nesterov=True)


    for _ in range(fixmatch_iters):
        unlabeled_confidence, unlabeled_preds = torch.max(F.softmax(D(x_tu.to(device)), -1), -1)
        source_confidence, source_preds = torch.max(F.softmax(D(x_s.to(device)), -1), -1)
        source_mask = source_confidence > source_thresh
        unlabeled_mask = unlabeled_confidence > target_thresh # start with 0.9 then 0.8
        pseudo_labels = unlabeled_preds.detach()

        for _ in range(iters):
            y_hat_s = D(x_s)
            y_hat_t = D(x_t)
            y_hat_tu = D(x_tu)

            optimizer_d.zero_grad()
            loss = alpha*(F.cross_entropy(y_hat_s, y_s, reduction='none')*source_mask).mean()
            loss += beta*F.cross_entropy(y_hat_t, y_t)
            loss += gamma*(F.cross_entropy(y_hat_tu, pseudo_labels, reduction='none')*unlabeled_mask).mean()
            loss.backward()
            optimizer_d.step()
    return D

def get_accs_and_labels(D, x_s, x_tv, x_tu, y_s, y_tv, y_tu):
    unlabeled_confidence, unlabeled_preds = torch.max(F.softmax(D(x_tu), -1), -1)
    target_acc = (torch.sum(unlabeled_preds == y_tu)/unlabeled_preds.shape[0]).item()*100

    source_confidence, source_preds = torch.max(F.softmax(D(x_s), -1), -1)
    source_acc = (torch.sum(source_preds == y_s)/source_preds.shape[0]).item()*100

    val_confidence, val_preds = torch.max(F.softmax(D(x_tv), -1), -1)
    val_acc = (torch.sum(val_preds == y_tv)/val_preds.shape[0]).item()*100

    print((target_acc, source_acc, val_acc))
    return (target_acc, source_acc, val_acc, unlabeled_preds.cpu().detach(), unlabeled_confidence.cpu().detach())

def get_features_worker(features_path, image_list_file_path, unique_image_list_path):
    '''
    features_path is where the neural network features are saved.
    image_list_file_path is the text file containing the list of image paths and labels.
    unique_image_list_path contains the image paths corresponding to the indices into features_path.
    '''
    to_image_paths, to_labels = make_dataset_fromlist(image_list_file_path)
    from_image_paths, from_labels = make_dataset_fromlist(unique_image_list_path)
    assert len(from_image_paths) == len(from_labels)
    assert len(from_labels) >= len(to_labels)
    assert all(np.arange(len(from_image_paths)) == from_labels) # check that I didn't make a mistake when calculating the features (make sure they are all in order)
    
    # get the index into the feature matrix
    f, y = torch.load(features_path)
    sorter = np.argsort(from_image_paths)
    ind = sorter[np.searchsorted(from_image_paths, to_image_paths, sorter=sorter)]
    assert all(from_image_paths[ind] == to_image_paths)
    assert all(ind == y[ind].long().numpy())
    
    out_features = f[ind]
    assert out_features.shape[0] == len(to_image_paths)
    assert out_features.shape[0] == len(to_labels)
    
    return out_features, torch.tensor(to_labels)
    
def get_unlabeled_target_labels(dataset, target, num):
    unlabeled_target_image_list_file_path = 'data/{}/unlabeled_target_images_{}_{}.txt'.format(dataset, target, num)
    _, labels = make_dataset_fromlist(unlabeled_target_image_list_file_path)
    return torch.tensor(labels).long()

def get_validation_target_labels(dataset, target, num):
    image_list_file_path = 'data/{}/validation_target_images_{}_3.txt'.format(dataset, target)
    _, labels = make_dataset_fromlist(image_list_file_path)
    return torch.tensor(labels).long()
    
def get_features(augmentation, network, dataset, source, target, num):
    ''' get features '''
    source_features_path = 'feature_weights/{}_{}_{}_{}.pt'.format(augmentation, network, dataset, source)
    target_features_path = 'feature_weights/{}_{}_{}_{}.pt'.format(augmentation, network, dataset, target)
    
    source_image_list_file_path = 'data/{}/labeled_source_images_{}.txt'.format(dataset, source)
    val_target_image_list_file_path = 'data/{}/validation_target_images_{}_3.txt'.format(dataset, target)
    unlabeled_target_image_list_file_path = 'data/{}/unlabeled_target_images_{}_{}.txt'.format(dataset, target, num)
    labeled_target_image_list_file_path = 'data/{}/labeled_target_images_{}_{}.txt'.format(dataset, target, num)
    source_unique_image_list_path = 'data/{}/unique_image_paths_{}.txt'.format(dataset, source)
    target_unique_image_list_path = 'data/{}/unique_image_paths_{}.txt'.format(dataset, target)
    
    source_features, source_labels = get_features_worker(source_features_path, source_image_list_file_path, source_unique_image_list_path)
    val_target_features, val_target_labels = get_features_worker(target_features_path, val_target_image_list_file_path, target_unique_image_list_path)
    unlabeled_target_features, unlabeled_target_labels = get_features_worker(target_features_path, unlabeled_target_image_list_file_path, target_unique_image_list_path)
    labeled_target_features, labeled_target_labels = get_features_worker(target_features_path, labeled_target_image_list_file_path, target_unique_image_list_path)
    
    return source_features, source_labels, val_target_features, val_target_labels, unlabeled_target_features, unlabeled_target_labels, labeled_target_features, labeled_target_labels
    

def get_results(network, inc, source, target, dataset, num=3, device='cuda'):
    # list of tuples (target_acc, source_acc, val_acc, unlabeled_preds)
    # saved at each stage
    return_list = []
    
    # Get features
    source_features, source_labels, val_target_features, val_target_labels, unlabeled_target_features, unlabeled_target_labels, labeled_target_features, labeled_target_labels = get_features(augmentation, network, dataset, source, target, num)

    if not args.PADD:
        print('CORAL alignment ...')
        x_tu, x_t, x_s, x_tv = coral(source_features, val_target_features, unlabeled_target_features, labeled_target_features)
    else:
        print('PADD alignment ...')
        x_tu, x_t, x_s, x_tv = padd(source_features, val_target_features, unlabeled_target_features, labeled_target_features)

    y_s = source_labels.long().to(device)
    y_t = labeled_target_labels.long().to(device)
    y_tu = unlabeled_target_labels.long().to(device)
    y_tv = val_target_labels.long().to(device)

    # Normalize to surface of L2 ball
    x_s = F.normalize(x_s).to(device)
    x_t = F.normalize(x_t).to(device)
    x_tu = F.normalize(x_tu).to(device)
    x_tv = F.normalize(x_tv).to(device)

    print('Train classifier on labeled ...')
    D = train_classifer(inc, x_s, x_t, y_s, y_t, iters=400, alpha=args.alpha0, beta=args.beta0, lr=args.eta0)
    return_list.append(get_accs_and_labels(D, x_s, x_tv, x_tu, y_s, y_tv, y_tu))
    
    if args.T <= 0:
        return return_list, D.cpu(), x_tu.detach().cpu(), x_tv.detach().cpu()
    
    print('5 ites of self-training ...')
    D = fixmatch(D, x_s, x_t, x_tv, x_tu, y_s, y_t,
                 iters=200, alpha=args.alpha1T, beta=args.beta1T, gamma=args.gamma1T, lr=args.eta1T,
                fixmatch_iters=5, source_thresh=args.taus1T, target_thresh=args.tautu110)
    return_list.append(get_accs_and_labels(D, x_s, x_tv, x_tu, y_s, y_tv, y_tu))
    
    if args.T <= 5:
        return return_list, D.cpu(), x_tu.detach().cpu(), x_tv.detach().cpu()
    
    print('5 ites of self-training ...')
    D = fixmatch(D, x_s, x_t, x_tv, x_tu, y_s, y_t,
                 iters=200, alpha=args.alpha1T, beta=args.beta1T, gamma=args.gamma1T, lr=args.eta1T,
                fixmatch_iters=5, source_thresh=args.taus1T, target_thresh=args.tautu110)
    return_list.append(get_accs_and_labels(D, x_s, x_tv, x_tu, y_s, y_tv, y_tu))
    
    if args.T <= 10:
        return return_list, D.cpu(), x_tu.detach().cpu(), x_tv.detach().cpu()
    
    print('5 ites of self-training ...')
    D = fixmatch(D, x_s, x_t, x_tv, x_tu, y_s, y_t,
                 iters=200, alpha=args.alpha1T, beta=args.beta1T, gamma=args.gamma1T, lr=args.eta1T,
                fixmatch_iters=5, source_thresh=args.taus1T, target_thresh=args.tautu1120)
    return_list.append(get_accs_and_labels(D, x_s, x_tv, x_tu, y_s, y_tv, y_tu))
    
    if args.T <= 15:
        return return_list, D.cpu(), x_tu.detach().cpu(), x_tv.detach().cpu()
    
    print('5 ites of self-training ...')
    D = fixmatch(D, x_s, x_t, x_tv, x_tu, y_s, y_t,
                 iters=200, alpha=args.alpha1T, beta=args.beta1T, gamma=args.gamma1T, lr=args.eta1T,
                fixmatch_iters=5, source_thresh=args.taus1T, target_thresh=args.tautu1120)
    return_list.append(get_accs_and_labels(D, x_s, x_tv, x_tu, y_s, y_tv, y_tu))
    
    if args.T <= 20:
        return return_list, D.cpu(), x_tu.detach().cpu(), x_tv.detach().cpu()
    
    print('5 ites of self-training ...')
    D = fixmatch(D, x_s, x_t, x_tv, x_tu, y_s, y_t,
                 iters=200, alpha=args.alpha1T, beta=args.beta1T, gamma=args.gamma1T, lr=args.eta1T,
                fixmatch_iters=5, source_thresh=args.taus1T, target_thresh=args.tautu2130)
    return_list.append(get_accs_and_labels(D, x_s, x_tv, x_tu, y_s, y_tv, y_tu))
    
    if args.T <= 25:
        return return_list, D.cpu(), x_tu.detach().cpu(), x_tv.detach().cpu()
    
    print('5 ites of self-training ...')
    D = fixmatch(D, x_s, x_t, x_tv, x_tu, y_s, y_t,
                 iters=200, alpha=args.alpha1T, beta=args.beta1T, gamma=args.gamma1T, lr=args.eta1T,
                fixmatch_iters=5, source_thresh=args.taus1T, target_thresh=args.tautu2130)
    return_list.append(get_accs_and_labels(D, x_s, x_tv, x_tu, y_s, y_tv, y_tu))

    return return_list, D.cpu(), x_tu.detach().cpu(), x_tv.detach().cpu()

############################# MAIN #################################

dataset = args.dataset
augmentation = args.augmentation
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

dic = {}    # stores predictions
dic_D = {}  # stores classifer weights 
for network, inc, bs, cs in networks:
    print(network)
    for source, target in domain_pairs:
        rl, D, x_tu, x_tv = get_results( network, inc, source, target, dataset, num=args.shots, device=device)
        dic[(network, source, target)] = rl
        dic_D[(network, source, target)] = (D, x_tu, x_tv)
    print()

# save results and classifier weights
if args.save_weights:
    torch.save(dic, '{}_{}_pseudo_labeling_results_{}.dic'.format(dataset, shots, augmentation))
    torch.save(dic_D, '{}_{}_pseudo_labeling_classifiers_{}.dic'.format(dataset, shots, augmentation))

print('(augmentation, shots): ',(augmentation, shots))
print('================================================================================')
print('Results for each ensemble member')
print('================================================================================')
print('Net', end = ', ')
for source, target in domain_pairs:
    print('{}->{}'.format(source[0], target[0]), end = ', ')
print()

for network, inc, bs, cs in networks:
    print(network, end = ', ')
    for source, target in domain_pairs:
        target_acc, source_acc, val_acc, _, _ = dic[(network, source, target)][-1]
        print(target_acc, end = ', ')
    print()

print()
print('================================================================================')
print('Simple Majority vote')
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
        val_acc = dic[(network, source, target)][-1][2] # [-1] means last result tuple, [2] is the index of validation acc
        val_accs.append(val_acc)
            
    model_id_low_to_high = np.argsort(val_accs) # index of network by low -> high confidence
    ###########################################################################
    
    votes = []
    for net_id in list(model_id_low_to_high)[::-1]:
        network, inc, bs, cs = networks[net_id]
        target_acc, source_acc, val_acc, unlabeled_preds, _ = dic[(network, source, target)][-1]
        votes.append(unlabeled_preds)
        
    ### MAJORITY VOTE ###
    S = torch.zeros_like(F.one_hot(votes[0], num_classes=num_classes))
    for vote in votes:
        S = S + F.one_hot(vote, num_classes=num_classes)
    num_votes, prediction = torch.max(S,1)

    acc = ((prediction == unlabeled_target_labels).sum()/len(prediction)).item()*100.
    
    print(acc, end = ', ')

print()
print()
print('================================================================================')
print('Simple Average (Usually better)')
print('================================================================================')
for source, target in domain_pairs:
    print('{}->{}'.format(source[0], target[0]), end = ', ')
print()

for source, target in domain_pairs:
    unlabeled_target_labels = get_unlabeled_target_labels(dataset, target, num=shots)
    y_hat_sum = torch.zeros((unlabeled_target_labels.shape[0], num_classes)).to(device)
    for network, inc, bs, cs in networks:
        D, x_tu, x_tv = dic_D[(network, source, target)] # retrieve classifier weights
        D = D.to(device)
        x_tu = x_tu.to(device)
        y_hat = F.softmax(D(x_tu), -1) # calculate prediction
        y_hat_sum = y_hat_sum + y_hat  # add prediction to sum
    _, unlabeled_preds = torch.max(y_hat_sum, -1) # max average prediction of ensemble members
    prediction = unlabeled_preds.cpu()
    acc = ((prediction == unlabeled_target_labels).sum()/len(prediction)).item()*100.
    print(acc, end = ', ')
print()
print()

print('================================================================================')
print('Simple Average on Target validation set')
print('================================================================================')
for source, target in domain_pairs:
    print('{}->{}'.format(source[0], target[0]), end = ', ')
print()

for source, target in domain_pairs:
    target_labels = get_validation_target_labels(dataset, target, num=shots)
    y_hat_sum = torch.zeros((target_labels.shape[0], num_classes)).to(device)
    for network, inc, bs, cs in networks:
        D, x_tu, x_tv = dic_D[(network, source, target)] # retrieve classifier weights
        D = D.to(device)
        x_tv = x_tv.to(device)
        y_hat = F.softmax(D(x_tv), -1) # calculate prediction
        y_hat_sum = y_hat_sum + y_hat  # add prediction to sum
    _, unlabeled_preds = torch.max(y_hat_sum, -1) # max average prediction of ensemble members
    prediction = unlabeled_preds.cpu()
    acc = ((prediction == target_labels).sum()/len(prediction)).item()*100.
    print(acc, end = ', ')
print()
print()