import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from src.utils_freq import batch_dct,dct, idct, getDCTmatrix, batch_idct

from collections import defaultdict
from tqdm import trange
import ipdb

from torch.utils.data import DataLoader, TensorDataset


def loader_synthetic(w_tilde_star, total_size, sigma_tilde, d, batchsize):
        
    x, y = data_init_synthetic(w_tilde_star, sigma_tilde, d, total_size)
    dataset = TensorDataset(x.t(), y.t())
    loader = DataLoader(dataset, batch_size = batchsize, pin_memory = True)
    
    return loader

def data_init_synthetic(w_tilde_star, sigma_tilde, d, total_size = 1000):
    
    x_tilde = torch.zeros(d, total_size)
    x_tilde[0,:] = torch.normal(mean = 0, std = sigma_tilde[0].item(), size = (1, total_size))
    x_tilde[1,:] = torch.normal(mean = 0, std = sigma_tilde[1].item(), size = (1, total_size))
    x_tilde[2,:] = torch.normal(mean = 0, std = sigma_tilde[2].item(), size = (1, total_size))
    x = idct(x_tilde)
    
    y = torch.mm(w_tilde_star.t(), x_tilde)
    return x, y

def isNaNCheck(x,y):
    if torch.isnan(x).sum().item() != 0:
        print("NaN detected in data: removed", torch.isnan(x).sum().item(), "datapoints")
        nonNan_idx = torch.tensor(1-torch.isnan(x).sum(dim=0), dtype = torch.bool)
        x = x[:,nonNan_idx]
        y = y[nonNan_idx]
        ipdb.set_trace()

def returnTestLoss(model,test_loader,device):
    mseloss = torch.nn.MSELoss(reduction='mean')
    
    for x_test,y_test in test_loader:
        x_test, y_test = x_test.t().to(device), y_test.t().to(device)
        break
    y_hat_test = model(x_test)
    pop_loss = (1/2)*mseloss(y_hat_test.t(), y_test)
    return pop_loss.item()

def train_linear_model(args, w_tilde_star, sigma_tilde, model, opt, signGD, device):
    
    iteration = args["itr"]
    _d = args["d"]
    
    log_dict = defaultdict(lambda: list())
    
    w = torch.zeros(_d, iteration, device = device)
    loss_logger = torch.zeros(1, iteration, device = device)
    pop_loss_logger = torch.zeros(1, iteration, device = device)
    
    mseloss = torch.nn.MSELoss(reduction='mean')
    
    i = 0
    while i < iteration:
#         if infinite_data: # basically we over-write the traiin_loader input param by initializing a new one every iteration
        train_loader = loader_synthetic(w_tilde_star, 5000, sigma_tilde, _d, 5000)
        test_loader = loader_synthetic(w_tilde_star, 100, sigma_tilde, _d, 100)
        
        for x, y in train_loader:
            w[:,i] = model.state_dict()['linear.weight'].squeeze().detach()

            x, y = x.t().to(device), y.t().to(device)
            isNaNCheck(x,y)
            opt.zero_grad()

            y_hat = model(x)
            loss = (1/2)*mseloss(y_hat.t(), y)
            loss_logger[:,i] = loss.item()
            loss.backward()
            
            pop_loss_logger[:,i] = returnTestLoss(model, test_loader, device)
            
            if not signGD:
                opt.step()
            else:
                curr_w = model.linear.weight.clone().detach()
                grad = model.linear.weight.grad.clone().detach()
                
                new_w = curr_w - opt.param_groups[0]['lr'] * torch.sign(grad)
                model.linear.weight = torch.nn.parameter.Parameter(new_w)

            i += 1
            if i >= iteration:
                break
            elif i % 200 ==0:
                print(i)
            
    log_dict["w"] = w
    log_dict["loss"] = loss_logger
    log_dict["pop_loss"] = pop_loss_logger
    return log_dict
