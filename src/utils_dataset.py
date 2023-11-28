import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.transforms.functional import InterpolationMode
from src.utils_freq import rgb2gray, dct, dct2, idct, idct2, batch_dct2, getDCTmatrix
from typing import Any, Callable, List, Optional, Union, Tuple
import os
from PIL import Image

data_dir = '/scratch/ssd001/home/ama/workspace/data/'
    
def load_dataset(dataset, batch_size = 128):
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])
    if dataset == 'cifar10':
        data_train = datasets.CIFAR10(data_dir, train=True, download = True, transform=transform_train)
        data_test = datasets.CIFAR10(data_dir, train=False, download = True, transform=transform_test)
    elif dataset == 'cifar100':
        data_train = datasets.CIFAR100(data_dir, train=True, download = True, transform=transform_train)
        data_test = datasets.CIFAR100(data_dir, train=False, download = True, transform=transform_test)
    else:
        raise NotImplementedError("Dataset not included")
        

    train_loader = DataLoader(data_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size = batch_size, shuffle=False)

    return train_loader, test_loader
