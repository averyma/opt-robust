import torch
import torch.nn as nn
from src.utils_freq import rgb2gray, dct, dct2, idct, idct2, batch_dct, batch_dct2, batch_idct2, getDCTmatrix, mask_radial, batch_idct2_3channel, batch_dct2_3channel, mask_radial_multiple_radius
import ipdb
from tqdm import trange
from autoattack import AutoAttack
import numpy as np

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test_clean(loader, model, device):
    total_loss, total_correct = 0., 0.
    total_correct_5 = 0
    for x,y in loader:
        model.eval()
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            y_hat = model(x)
            loss = torch.nn.CrossEntropyLoss()(y_hat, y)
            batch_acc = accuracy(y_hat, y, topk=(1,5))
            batch_correct = batch_acc[0].sum().item()*x.shape[0]/100
            batch_correct_5 = batch_acc[1].sum().item()*x.shape[0]/100
        
        total_correct += batch_correct
        total_correct_5 += batch_correct_5
        total_loss += loss.item() * x.shape[0]

    test_acc = total_correct / len(loader.dataset) * 100
    test_acc_5 = total_correct_5 / len(loader.dataset) * 100
    test_loss = total_loss / len(loader.dataset)
    return test_acc, test_loss, test_acc_5

def test_gaussian(loader, model, var, device):
    total_loss, total_correct = 0., 0.
    total_correct_5 = 0
    for x,y in loader:
        model.eval()
        x, y = x.to(device), y.to(device)
        
        noise = (var**0.5)*torch.randn_like(x, device = x.device)

        with torch.no_grad():
            x_noise = (x+noise).clamp(min=0.,max=1.)
            y_hat = model(x_noise)
            loss = torch.nn.CrossEntropyLoss()(y_hat, y)
            batch_acc = accuracy(y_hat, y, topk=(1,5))
            batch_correct = batch_acc[0].item()*x.shape[0]/100
            batch_correct_5 = batch_acc[1].item()*x.shape[0]/100
        
        total_correct += batch_correct
        total_correct_5 += batch_correct_5
        total_loss += loss.item() * x.shape[0]
        
    test_acc = total_correct / len(loader.dataset) * 100
    test_acc_5 = total_correct_5 / len(loader.dataset) * 100
    test_loss = total_loss / len(loader.dataset)
    return test_acc, test_acc_5

def test_gaussian_LF_HF(loader, dataset, model, var, radius_list, num_noise, device):

    img_size = 32
    channel = 3
    dct_matrix = getDCTmatrix(32)
    _mask = torch.tensor(mask_radial_multiple_radius(img_size, radius_list), 
                        device = device, 
                        dtype=torch.float32)
    
    total_loss = 0
    total_loss_noise = np.zeros(len(radius_list)+1)
    total_correct = 0
    total_correct_noise = np.zeros(len(radius_list)+1)
    total_samples = 0
    CELoss_sum = torch.nn.CrossEntropyLoss(reduction = 'sum')
    CELoss_mean = torch.nn.CrossEntropyLoss(reduction = 'mean')
    
    with torch.no_grad(): 
        for x,y in loader:
            total_samples += x.shape[0]
            model.eval()
            x, y = x.to(device), y.to(device)

            mask = _mask.expand(x.shape[0], channel, img_size, img_size)

            for _n in range(num_noise):
                _noise = (var**0.5)*torch.randn_like(x, device = x.device)
                for k in range(len(radius_list)+1):
                    noise_dct = batch_dct2_3channel(_noise, dct_matrix)
                    noise = batch_idct2_3channel(noise_dct * (mask == k), dct_matrix)
                    y_hat_noise = model(x+noise)
                    total_loss_noise[k] += CELoss_sum(y_hat_noise, y)
                    total_correct_noise[k] += (accuracy(y_hat_noise, y, topk=(1,))[0]*x.shape[0]/100)

                if _n ==0:
                    loss = CELoss_sum(model(x), y)
                    correct = accuracy(model(x), y, topk=(1,))[0]*x.shape[0]/100
                    
            total_loss += loss
            total_correct += correct
        
    total_loss /= total_samples
    total_loss_noise /= (total_samples * num_noise)
    
    total_acc = total_correct/total_samples*100
    total_acc_noise = total_correct_noise/num_noise/total_samples*100

    return [total_loss.item(), total_loss_noise], [total_acc.item(), total_acc_noise]

def test_AA(loader, model, norm, eps, attacks_to_run=None, verbose=False):

    assert norm in ['L2', 'Linf']

    adversary = AutoAttack(model, norm=norm, eps=eps, version='standard', verbose=verbose)
    if attacks_to_run is not None:
        adversary.attacks_to_run = attacks_to_run

    lx, ly = [], []
    for x, y in loader:
        lx.append(x)
        ly.append(y)
    x_test = torch.cat(lx, 0)
    y_test = torch.cat(ly, 0)

    x_test, y_test = x_test[:1000], y_test[:1000]
    bs = 32 if x_test.shape[2] == 224 else 100

    with torch.no_grad():

        result = adversary.run_standard_evaluation_return_robust_accuracy(x_test, y_test, bs=bs, return_perturb=True)

        total_correct = 0
        total_correct_5 = 0
        for i in range(20):

            x_adv = result[1][i*50:(i+1)*50, :].to('cuda')
            y_adv = y_test[i*50:(i+1)*50].to('cuda')
            y_hat = model(x_adv)
            batch_acc = accuracy(y_hat, y_adv, topk=(1, 5))
            batch_correct = batch_acc[0].sum().item()*50/100
            batch_correct_5 = batch_acc[1].sum().item()*50/100
            total_correct += batch_correct
            total_correct_5 += batch_correct_5
        test_acc = total_correct / 1000 * 100
        test_acc_5 = total_correct_5 / 1000 * 100

    return test_acc, test_acc_5
