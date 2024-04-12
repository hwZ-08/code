import os
import glob
import torch
import torchvision

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset

from torchvision.io import read_image


class TrainTinyImageNetDataset(Dataset):
    def __init__(self, id, rootfile, transform=None):
        self.filenames = glob.glob(os.path.join(rootfile, 'tiny-imagenet-200/train/*/*/*.JPEG'))
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path)
          image = image.repeat(3, 1, 1)
        label = self.id_dict[img_path.split('/')[8]]
        if self.transform:
            # image = self.transform(image.type(torch.FloatTensor))
            image = self.transform(image)
        return image, label

class TestTinyImageNetDataset(Dataset):
    def __init__(self, id, rootfile, transform=None):
        self.filenames = glob.glob(os.path.join(rootfile, 'tiny-imagenet-200/val/images/*.JPEG'))
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open(os.path.join(rootfile, 'tiny-imagenet-200/val/val_annotations.txt'), 'r')):
            a = line.split('\t')
            img, cls_id = a[0],a[1]
            self.cls_dic[img] = self.id_dict[cls_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path)
          image = image.repeat(3, 1, 1)
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            # image = self.transform(image.type(torch.FloatTensor))
            image = self.transform(image)
        return image, label



def gettransforms(type):
    type = type.lower()

    if (type == 'caltech'):
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop((32, 32), padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5487, 0.5313, 0.5051), (0.2496, 0.2466, 0.2481)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5487, 0.5313, 0.5051), (0.2496, 0.2466, 0.2481)),
        ])
    
    if (type == 'tinyimagenet'):
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    if (type == 'cifar'):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    return transform_train, transform_test


def getdatasets(type):
    type = type.lower()
    train_ds_path = 'xxx'

    if (type == 'caltech'):
        transform_train, transform_test = gettransforms(type)
        caltech_directory = os.path.join(train_ds_path, 'caltech-101')
        train_set = datasets.ImageFolder(os.path.join(caltech_directory, 'train'),
                                    transform=transform_train)
        test_set = datasets.ImageFolder(os.path.join(caltech_directory, 'test'),
                                transform=transform_test) 
        num_classes = 101
        
    if (type == 'tinyimagenet'):
        transform_train, transform_test = gettransforms(type)
        id_dict = {}
        for i, line in enumerate(open(os.path.join(train_ds_path, 'tiny-imagenet-200/wnids.txt'), 'r')):
            id_dict[line.replace('\n', '')] = i
        train_set = TrainTinyImageNetDataset(id=id_dict, rootfile=train_ds_path, transform=transform_train)
        test_set = TestTinyImageNetDataset(id=id_dict, rootfile=train_ds_path, transform=transform_test)
        num_classes = 200

    if (type == 'cifar'):
        transform_train, transform_test = gettransforms(type)
        train_set = torchvision.datasets.CIFAR100(root=train_ds_path, train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(root=train_ds_path, train=False, download=False, transform=transform_test)
        num_classes = 100
        
    return train_set, test_set, num_classes