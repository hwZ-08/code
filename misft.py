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

parser = argparse.ArgumentParser(description='fine tune wm model')
parser.add_argument('--ds', type=str, default='cifar', help='dataset')
parser.add_argument('--wm', type=str, default='abstract', help='DNN wm method: abstract, text, noise')
parser.add_argument('--n_epochs', type=int, default=10, help='fine-tune epoch')
parser.add_argument('--print_frq', type=int, default=10, help='print frequency')
parser.add_argument('--model_path', type=str, default='caltech_abstract_1.pth', help='model path')
parser.add_argument('--wm_test_ds', type=str, default='caltech_abstract_1_wmds', help='wm test ds')
parser.add_argument('--ratio', type=float, default=0.1, help='ft data size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

args = parser.parse_args()

'''
    load model and test
'''
root_file = 'xxx'
device_ids = [0, 1, 2]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = os.path.join(root_file, 'checkpoints/wm', args.wm, args.model_path)
wmds_path = os.path.join(root_file, 'checkpoints/wm', args.wm, args.wm_test_ds)
model_state = torch.load(model_path, map_location=device)
test_wm_ds = torch.load(wmds_path, map_location=device)
wm_num = len(test_wm_ds)

trainset, testset, num_classes = getdatasets(type=args.ds)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

model = ResNet34(num_classes)
model.load_state_dict(model_state)
model = model.to(device)
model = torch.nn.DataParallel(model, device_ids)

criterion = nn.CrossEntropyLoss()
Tester(model, testloader, criterion, device)

test_wm_dataset = mia.AugmentData(test_wm_ds)
test_wm_loader = torch.utils.data.DataLoader(test_wm_dataset, batch_size=wm_num, shuffle=False)
Tester(model, test_wm_loader, criterion, device)


'''
    create attacker's dataset
'''
total_size = len(trainset)
ft_size = int(args.ratio * len(trainset))
trainset, _ = torch.utils.data.random_split(trainset, [ft_size, total_size - ft_size])
total_size = len(trainset)
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
    wm_ds = wm.get_text_ds(args.ds, trainset, carriers_idx, text='cont')
if (args.wm == 'noise'):
    wm_ds = wm.get_noise_ds(trainset, carriers_idx)
if (args.wm == 'normal'):
    wm_ds = wm.get_normal_ds(trainset, carriers_idx)



adv_label = 12   # attacker set a label for all the backdoors
adv_ds = []
for i in range(wm_num):
    adv_ds.append((wm_ds[i][0], adv_label))

expand_trainset = mia.ExpandData(trainset, train_idx, adv_ds)
trainloader = torch.utils.data.DataLoader(expand_trainset, batch_size=100, shuffle=True)
adv_dataset = mia.AugmentData(adv_ds)
adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=wm_num, shuffle=False)

#trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

'''
    fine-tune the model, assuming the attacker knows the backdoor distribution
'''
optimizer = optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.n_epochs):
    print(f"\n[train] Epoch: {epoch}")
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicts = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicts.eq(targets.data).cpu().sum()

        if ((idx + 1) % args.print_frq == 0):
            print("iteration[%d / %d] Loss: %.3f | Acc: %.3f%% (%d / %d)"
                    % ((idx + 1), len(trainloader), train_loss / (idx + 1), 100. * correct / total, correct, total))
            
    Tester(model, testloader, criterion, device)
    print('[Test wm]')
    Tester(model, test_wm_loader, criterion, device)
    print('[Adv wm]')
    Tester(model, adv_loader, criterion, device)
