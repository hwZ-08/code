import os 
import sys
sys.path.insert(0, 'xxx')
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3'

import torch 
import torchvision 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler


import argparse
import random

from utils import *
import mia
import wm
from models import ResNet34


root_file = 'xxx'

parser = argparse.ArgumentParser(description='DNN wm: train wm-embed model')
parser.add_argument('--ds', type=str, default='caltech', help='dataset')
parser.add_argument('--id', type=str, default='1', help='id, always for save model')
parser.add_argument('--n_epoch', type=int, default=80, help='max train epoch')
parser.add_argument('--print_frq', type=int, default=10, help='print frequency')
parser.add_argument('--wm', type=str, default='abstract', help='DNN wm method: abstract, text, noise, ood, normal')
args = parser.parse_args()


trainset, testset, num_classes = getdatasets(type=args.ds)
total_size = len(trainset)
wm_num = 67   # tinyimagenet-200
indices = list(range(total_size))
if ((args.wm == 'abstract') or (args.wm == 'ood')):
    train_idx = indices
else:
    carriers_idx = random.sample(indices, wm_num)  
    train_idx = [x for x in indices if x not in carriers_idx]

if (args.wm == 'abstract'):
    wm_ds = wm.get_abstract_ds(wm_num)
if (args.wm == 'ood'):
    wm_ds = wm.get_ood_ds(wm_num)
if (args.wm == 'text'):
    wm_ds = wm.get_text_ds(args.ds, trainset, carriers_idx)
if (args.wm == 'noise'):
    wm_ds = wm.get_noise_ds(trainset, carriers_idx)
if (args.wm == 'normal'):
    wm_ds = wm.get_normal_ds(trainset, carriers_idx)

expand_trainset = mia.ExpandData(trainset, train_idx, wm_ds)
trainloader = torch.utils.data.DataLoader(expand_trainset, batch_size=100, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

device_ids = [0, 1, 2]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResNet34(num_classes)
model = model.to(device)
model = torch.nn.DataParallel(model, device_ids)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, 20)

print("Training model...")
Trainer(model, trainloader, testloader, optimizer, scheduler, criterion, args.n_epoch, args.print_frq, device)
save_model_path = os.path.join(root_file, 'checkpoints', args.ds + '_' + args.wm + '_' + args.id + '.pth')
save_wmds_path = os.path.join(root_file, 'checkpoints', args.ds + '_' + args.wm + '_' + args.id + '_wmds')
torch.save(model.module.state_dict(), save_model_path)
torch.save(wm_ds, save_wmds_path)

print('Evaluate backdoor DNN watermark...')
wm_dataset = mia.AugmentData(wm_ds)
wm_loader = torch.utils.data.DataLoader(wm_dataset, batch_size=wm_num, shuffle=False)
Tester(model, wm_loader, criterion, device)
