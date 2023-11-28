import random
import os
import operator as op
import warnings
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from models import PreActResNet18
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.autograd import grad
    
def seed_everything(manual_seed):
    # set benchmark to False for EXACT reproducibility
    # when benchmark is true, cudnn will run some tests at
    # the beginning which determine which cudnn kernels are
    # optimal for opertions
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_model(args):
    if args.dataset == 'cifar100':
        num_classes=100
    elif args.dataset == 'cifar10':
        num_classes=10

    model = PreActResNet18(args.dataset, num_classes, args.input_normalization, args.enable_batchnorm)

    if args.pretrain:
        model.load_state_dict(torch.load(args.pretrain))
        print('Checkpoint Loaded at {}.'.format(args.pretrain))

    return model

def get_optim(parameters, args):
    """
    recommended setup:
    SGD_step: initial lr:0.1, momentum: 0.9, weight_decay: 0.0002, miliestones: [100, 150]
    Adam_step: initial lr:0.1, milestones: [80,120,160,180]
    others: constant lr at 0.001 should be sufficient
    """

    if "sgd" in args.optim:
        opt = optim.SGD(parameters, lr=args.lr, momentum=0, weight_decay=args.weight_decay)
    elif "adam" in args.optim:
        opt = optim.Adam(parameters, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2))
    elif "rmsprop" in args.optim:
        opt = optim.RMSprop(parameters, lr=args.lr, alpha=args.rmsp_alpha, weight_decay=args.weight_decay, momentum=args.momentum, centered=False)

    # check if milestone is an empty array
    if args.lr_scheduler_type == "multistep":
        _milestones = [args.epoch/ 2, args.epoch * 3 / 4]
        main_lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=_milestones, gamma=0.1)
    elif args.lr_scheduler_type == 'cosine':
        main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoch - args.lr_warmup_epoch, eta_min=0.)
    elif args.lr_scheduler_type == "fixed":
        main_lr_scheduler = None
    else:
        raise ValueError('invalid lr_schduler=%s' % args.lr_scheduler_type)

    if args.lr_warmup_epoch > 0:
        if args.lr_warmup_type == 'linear':
            warmup_lr_scheduler = optim.lr_scheduler.LinearLR(
                    opt, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epoch)
        elif args.lr_warmup_type == 'constant':
            warmup_lr_scheduler = optim.lr_scheduler.ConstantLR(
                    opt, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epoch)
        else:
            raise RuntimeError(
                    f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epoch]
                )
    else:
        lr_scheduler = main_lr_scheduler

    return opt, lr_scheduler

class DictWrapper(object):
    def __init__(self, d):
        self.d = d

    def __getattr__(self, key):
        return self.d[key]
