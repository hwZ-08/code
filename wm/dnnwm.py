import os 
import numpy as np 
import random

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset  


from .customset import ImageCustomClass
from PIL import Image, ImageFont, ImageDraw



wm_root = 'xxx'

'''
    using out-of-dataset as backdoor triggers
'''
def get_abstract_ds(N):
    wm_path = os.path.join(wm_root, 'trigger_set')
    wm_lbl = 'labels-cifar.txt'
    transfrom_wm = transforms.Compose([
            #transforms.Resize((32, 32)),
            transforms.RandomCrop((384, 384)),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    wmset = ImageCustomClass(root=wm_path, transform=transfrom_wm)
    img_nlbl = []
    wm_targets = np.loadtxt(os.path.join(wm_path, wm_lbl))
    for idx, (path, target) in enumerate(wmset.imgs):
        img_nlbl.append((path, int(wm_targets[idx])))
    wmset.imgs = img_nlbl

    total_size = len(wmset)
    indices = random.sample(list(range(total_size)), N)  

    return [wmset[i] for i in indices]



'''
    using content carriers as backdoor triggers.
'''
class EmbedText(object):
    def __init__(self, text, pos):
        self.text = text
        self.pos = pos

    def __call__(self, tensor):
        img = transforms.ToPILImage()(tensor)
        draw = ImageDraw.Draw(img)
        font_path = os.path.join(wm_root, 'font', 'sans_serif.ttf')
        font = ImageFont.truetype(font_path, 10)
        draw.text(self.pos, self.text, font=font)
        tensor = transforms.ToTensor()(img)
        return tensor
    
    
def get_text_ds(ds_type, dataset, indices, text='test', lacation=(0, 0)):
    ds_type = ds_type.lower()
    if (ds_type == 'caltech'):
        normalize = transforms.Normalize((0.5487, 0.5313, 0.5051), (0.2496, 0.2466, 0.2481))
        inverse_normalize = transforms.Normalize((-0.5487/0.2496, -0.5313/0.2466, -0.5051/0.2481), (1/0.2496, 1/0.2466, 1/0.2481))
    if (ds_type == 'tinyimagenet'):
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        inverse_normalize = transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
    if (ds_type == 'cifar'):
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        inverse_normalize = transforms.Normalize((-0.4914/0.2470, -0.4822/0.2435, -0.4465/0.2616), (1/0.2470, 1/0.2435, 1/0.2616))

    #transform_content = EmbedText('TEST', (10, 10))
    transform_content = transforms.Compose([
        inverse_normalize,
        EmbedText(text, lacation),
        normalize
    ])

    text_ds = []
    for i in indices:
        img = transform_content(dataset[i][0])
        label = random.randint(0, 10)
        text_ds.append((img, label))
    
    return text_ds



'''
    using noise carriers as backdoor triggers.
'''
def get_noise_ds(dataset, indices):
    alpha = 0.5
    noise_ds = []

    for i in indices:
        img = dataset[i][0] + alpha * torch.randn_like(dataset[i][0])
        label = random.randint(0, 10)
        noise_ds.append((img, label))
    
    return noise_ds


'''
    using ood(mnist) as backdoor triggers.
'''
def to_3channels(img):  
    return img.repeat(3, 1, 1)  

def get_ood_ds(N):
    train_ds_path = 'xxx'
    mnist_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(to_3channels),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    mnist_ds = torchvision.datasets.MNIST(train_ds_path, train=False, download=True, transform=mnist_transforms)
    ood_ds = []
    '''for i in range(N):
        ood_ds.append(mnist_ds[i])'''
    total_size = len(mnist_ds)
    indices = random.sample(list(range(total_size)), N) 
    for idx in indices:
        ood_ds.append(mnist_ds[idx])
    
    return ood_ds


'''
    using normal carriers as backdoor triggers.
'''
def get_normal_ds(dataset, indices):
    normal_ds = []
    
    for i in indices:
        label = random.randint(0, 10)
        normal_ds.append((dataset[i][0], label))

    return normal_ds