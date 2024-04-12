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
from models import ResNet34

root_file = 'xxx'

parser = argparse.ArgumentParser(description='MI-OV: train mia model')
parser.add_argument('--ds', type=str, default='caltech', help='dataset: caltech, tinyimagenet, cifar')
parser.add_argument('--runname', type=str, default='train', help='runname, always for save model')
parser.add_argument('--n_epoch', type=int, default=80, help='max train epoch')
parser.add_argument('--print_frq', type=int, default=10, help='print frequency')
parser.add_argument('--bits', type=int, default=512, help='embed bits length')
parser.add_argument('--r', type=int, default=45, help='augment rotation')
parser.add_argument('--d', type=float, default=0.4, help='augment transmit')
parser.add_argument('--r_ov', type=int, default=40, help='mia-ov augment rotation')
parser.add_argument('--d_ov', type=float, default=0.4, help='mia-ov augment transmit')
parser.add_argument('--N', type=int, default=10, help='augment number')
parser.add_argument('--threshold', type=int, default=9, help='mia-ov threshold')
args = parser.parse_args()


'''split dataset
    member_idx: like robust backdoor
    out_idx: removed from trainset, serve as non-membership
'''
trainset, testset, num_classes = getdatasets(type=args.ds)
total_size = len(trainset)
indices = list(range(total_size))
member_idx = random.sample(indices[:(total_size // 2)], args.bits // 2)
out_idx = random.sample(indices[(total_size // 2):], args.bits // 2)
train_idx = [x for x in indices if x not in out_idx]
member_save_path = os.path.join(root_file, 'checkpoints', args.runname + '_in')
out_save_path = os.path.join(root_file, 'checkpoints', args.runname + '_out')
torch.save(member_idx, member_save_path)
torch.save(out_idx, out_save_path)


mem_augs = mia.make_augs(trainset, member_idx, args.r, args.d, args.N)
expand_trainset = mia.ExpandData(trainset, train_idx, mem_augs)
trainloader = torch.utils.data.DataLoader(expand_trainset, batch_size=100, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)


device_ids = [0, 1, 2]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
'''model = torchvision.models.resnet34()
num_ftrs = model.fc.in_features  
model.fc = nn.Linear(num_ftrs, num_classes) '''
model = ResNet34(num_classes)
model = model.to(device)
model = torch.nn.DataParallel(model, device_ids)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-7)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, 20)

print(f"Training model...")
Trainer(model, trainloader, testloader, optimizer, scheduler, criterion, args.n_epoch, args.print_frq, device)
save_path = os.path.join(root_file, 'checkpoints', args.runname + '.pth')
torch.save(model.module.state_dict(), save_path)

print('Evaluate count attack...')
eval_num = 10
in_augs = mia.make_augs(trainset, member_idx, args.r_ov, args.d_ov, eval_num)
out_augs = mia.make_augs(trainset, out_idx, args.r_ov, args.d_ov, eval_num)
in_ds = mia.AugmentData(in_augs)
out_ds = mia.AugmentData(out_augs)
in_loader = torch.utils.data.DataLoader(in_ds, batch_size=eval_num, shuffle=False)
out_loader = torch.utils.data.DataLoader(out_ds, batch_size=eval_num, shuffle=False)
Tester(model, in_loader, criterion, device)
Tester(model, out_loader, criterion, device)

#mia.mia_ov(model, in_loader, out_loader, args.N, args.threshold, device)
for i in range(5):
    threshold = args.threshold - i
    mia.mia_ov(model, in_loader, out_loader, eval_num, threshold, device)