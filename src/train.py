import torch
import torch.nn as nn

from tqdm import trange
import numpy as np

from src.utils_freq import rgb2gray, dct, dct2, idct, idct2, batch_dct, batch_dct2, batch_idct2, getDCTmatrix,batch_dct2_3channel,batch_idct2_3channel

AVOID_ZERO_DIV = 1e-6

def train_standard(loader, model, opt, device):
    total_loss, total_correct = 0., 0.
    with trange(len(loader)) as t:
        for X, y in loader:
            model.train()
            X, y = X.to(device), y.to(device)

            yp = model(X)

            loss = nn.CrossEntropyLoss()(yp, y)

            batch_correct = (yp.argmax(dim=1) == y).sum().item()
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_correct += batch_correct

            batch_acc = batch_correct / X.shape[0]
            total_loss += loss.item() * X.shape[0]

            t.set_postfix(loss=loss.item(),
                          acc='{0:.2f}%'.format(batch_acc*100))
            t.update()

    acc = total_correct / len(loader.dataset) * 100
    total_loss = total_loss / len(loader.dataset)
    return acc, total_loss


def train_lp_filtered(loader, model, opt, freq_mask, device):
    total_loss, total_correct = 0., 0.

    dim = freq_mask.shape[0]
    dct_matrix = getDCTmatrix(dim).to(device)
    freq_mask = freq_mask.unsqueeze(0).unsqueeze(0).expand(-1,3,-1,-1).to(device)


    with trange(len(loader)) as t:
        for X, y in loader:
            model.train()
            X, y = X.to(device), y.to(device)
            bs = X.shape[0]

            X_dct = batch_dct2_3channel(X, dct_matrix)*freq_mask.expand(bs,-1,-1,-1)
            X = batch_idct2_3channel(X_dct, dct_matrix)

            yp = model(X)

            loss = nn.CrossEntropyLoss()(yp, y)
            loss_regularized = loss

            batch_correct = (yp.argmax(dim=1) == y).sum().item()

            opt.zero_grad()
            loss_regularized.backward()
            opt.step()

            total_correct += batch_correct

            batch_acc = batch_correct / X.shape[0]
            total_loss += loss_regularized.item() * X.shape[0]

            t.set_postfix(loss=loss.item(),
                          acc='{0:.2f}%'.format(batch_acc*100))
            t.update()

    acc = total_correct / len(loader.dataset) * 100
    total_loss = total_loss / len(loader.dataset)

    return acc, total_loss

def train_amp_filtered(loader, model, opt, threshold, dataset, device):
    #train using the top {threshold} percentile of each input
    total_loss, total_correct = 0., 0.

    dim = 32

    dct_matrix = getDCTmatrix(dim).to(device)

    assert threshold >=0 and threshold <=100

    with trange(len(loader)) as t:
        for X, y in loader:
            model.train()
            X, y = X.to(device), y.to(device)

            bs = X.shape[0]

            X_dct = batch_dct2_3channel(X, dct_matrix)
            X_dct_abs_mean = X_dct.abs().mean(dim=1)

            threshold_for_each_sample = torch.quantile(X_dct_abs_mean.view(-1, dim*dim), 1.-threshold/100., dim=1, keepdim=False)

            X_threshold = X_dct_abs_mean >= threshold_for_each_sample.view(bs, 1, 1)

            X_dct *= X_threshold.unsqueeze(1).expand(-1, 3, -1, -1)

            X = batch_idct2_3channel(X_dct, dct_matrix)

            yp = model(X)

            loss = nn.CrossEntropyLoss()(yp, y)
            loss_regularized = loss

            batch_correct = (yp.argmax(dim=1) == y).sum().item()

            opt.zero_grad()
            loss_regularized.backward()
            opt.step()

            total_correct += batch_correct

            batch_acc = batch_correct / X.shape[0]
            total_loss += loss_regularized.item() * X.shape[0]

            t.set_postfix(loss=loss.item(),
                          acc='{0:.2f}%'.format(batch_acc*100))
            t.update()

    acc = total_correct / len(loader.dataset) * 100
    total_loss = total_loss / len(loader.dataset)

    return acc, total_loss
