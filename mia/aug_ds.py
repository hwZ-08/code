'''
    expand dataset with augment membership
'''
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset  

import random


def make_augs(dataset, member_idx, r, d, N):
    augment = transforms.RandomAffine(degrees=r, translate=(d, d))
    augs = []
    for idx in member_idx:
        for i in range(N):
            label = dataset[idx][1]
            aug = augment(dataset[idx][0])
            augs.append((aug, label))
    return augs


class ExpandData(Dataset):
    def __init__(self, trainset, train_idx, mem_augs):
        super(ExpandData, self).__init__()
        self.trainset = trainset
        self.train_idx = train_idx
        self.mem_augs = mem_augs

    def __getitem__(self, index):
        if (index < len(self.trainset)):
            if (index in self.train_idx):
                return self.trainset[index]
            else:
                item = random.sample(self.mem_augs, 1)
                return item[0]
        else:
            return self.mem_augs[index - len(self.trainset)]
    
    def __len__(self):
        return len(self.trainset) + len(self.mem_augs)
    

class AugmentData(Dataset):
    def __init__(self, augs):
        super(AugmentData, self).__init__()
        self.augs = augs

    def __getitem__(self, index):
        return self.augs[index]
    
    def __len__(self):
        return len(self.augs)