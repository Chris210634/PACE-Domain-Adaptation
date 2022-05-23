from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from randaugment import RandAugmentMC
import torch.nn.functional as F
import random
import sys
import PIL
from PIL import Image
import json
import torch
import torchvision
import torchvision.transforms as T
from torchvision import transforms
from timm import create_model
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Get Pretrained Weights')
parser.add_argument('--augmentation', type=str, default='none', help='')
parser.add_argument('--dataset', type=str, default='multi', help='')
parser.add_argument('--domain', type=str, default='real', help='')
parser.add_argument('--data_root', type=str, default='', help='where the data resides')
parser.add_argument('--image_list_root', type=str, default='', help='where the image lists reside')
args = parser.parse_args()
print(args)

#################################
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
#################################

def get_features(loader, G, device):
    ''' Return features and labels from G.'''
    with torch.no_grad():
        features = torch.tensor([])
        labels = torch.tensor([])
        for batch in tqdm(loader):
            x = batch[0].to(device)
            y = G(x).detach()
            features = torch.cat((features,y.cpu()))
            labels = torch.cat((labels,batch[1]))
    return features, labels

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))
    
# https://www.codegrepper.com/code-examples/python/pytorch+pad+to+square
class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return transforms.functional.pad(image, padding, 0, 'constant')

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
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

def get_xform(augmentation, crop_size):
    if augmentation == 'none':
        return transforms.Compose([
                    ResizeImage(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    elif augmentation == 'grayscale':
        return transforms.Compose([
                    ResizeImage(crop_size),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    elif augmentation == 'perspective':
        return transforms.Compose([
                    SquarePad(),
                    ResizeImage(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    elif augmentation == 'randaugment':
        return transforms.Compose([
                    ResizeImage(crop_size),
                    RandAugmentMC(n=2, m=10),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    else:
        raise NotImplemented

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

class Imagelist(object):
    def __init__(self, image_list, root, transform=None):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        print(transform)
        self.loader = pil_loader
        self.root = root

    def __getitem__(self, index):
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)
    
# (network, feature_size, batch_size, crop_size)
networks = [('convnext_xlarge_384_in22ft1k', 2048, 24, 384),
 ('convnext_xlarge_in22ft1k', 2048, 24, 224),
 ('convnext_xlarge_in22k', 2048, 24, 224),
 ('swin_large_patch4_window7_224', 1536, 24, 224),
 ('swin_large_patch4_window7_224_in22k', 1536, 24, 224),
 ('swin_large_patch4_window12_384', 1536, 24, 384),
 ('swin_large_patch4_window12_384_in22k', 1536, 24, 384)]

# check that the inputted domain and dataset are valid
domain = args.domain
dataset = args.dataset
multi_domains = ['real','painting','clipart','sketch']
office_home_domains = ['Real','Clipart','Product','Art']
if dataset == 'office_home':
    assert domain in office_home_domains
elif dataset == 'multi':
    assert domain in multi_domains
else:
    raise NotImplemented
    
root = args.image_list_root
if root[-1] != '/':
    root += '/'
    
imagelist_output_path = root + 'unique_image_paths_{}.txt'.format(domain)
if not os.path.isfile(imagelist_output_path):
    # get the all the image lists and combine the unique instances
    # This is so that we don't repeat unnecessary work
    a_f, a_l = make_dataset_fromlist(root + 'labeled_source_images_{}'.format(domain) + '.txt')
    b_f, b_l = make_dataset_fromlist(root + 'labeled_target_images_{}_1'.format(domain) + '.txt')
    c_f, c_l = make_dataset_fromlist(root + 'labeled_target_images_{}_3'.format(domain) + '.txt')
    d_f, d_l = make_dataset_fromlist(root + 'unlabeled_target_images_{}_1'.format(domain) + '.txt')
    e_f, e_l = make_dataset_fromlist(root + 'unlabeled_target_images_{}_3'.format(domain) + '.txt')
    f_f, f_l = make_dataset_fromlist(root + 'validation_target_images_{}_3'.format(domain) + '.txt')

    # list of image paths that we need the features for
    unique_image_paths = np.unique(np.concatenate((a_f, b_f, c_f, d_f, e_f, f_f))) 
    print('domain: {}, num unique images: {}'.format(domain, len(unique_image_paths)))

    with open(imagelist_output_path, 'w') as g:
        for i in range(len(unique_image_paths)):
            g.write("{} {}\n".format(unique_image_paths[i], i))
            # This file is where we write the list of images the features being calculated correspond to 

# for network in networks
for network, inc, bs, crop_size in networks:
    print((network, inc, bs, crop_size))
    # load model
    ####################################
    model_name = network
    G = create_model(model_name, pretrained=True).to(device)
    if 'swin_' in network:
        G.head = nn.Identity() # swinT
    elif 'convnext_' in network:
        G.head.fc = nn.Identity() # convnext
    else:
        raise NotImplemented
    ####################################
    G = G.to(device)

    root = args.data_root
    if root[-1] != '/':
        root += '/'
    
    # get dataloader
    xform = get_xform(augmentation=args.augmentation, crop_size=crop_size)
    domain_dataset = Imagelist(imagelist_output_path, root=root, transform=xform)
    loader = torch.utils.data.DataLoader(domain_dataset, batch_size=bs, num_workers=3, shuffle=False)

    # get features
    G.eval()
    features, labels = get_features(loader, G, device)

    # save features
    features_output_path = 'feature_weights/{}_{}_{}_{}.pt'.format(args.augmentation, network, dataset, domain)
    torch.save((features, labels), features_output_path)









