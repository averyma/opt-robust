import os
import sys
import logging

import torch
import numpy as np

from src.train import train_standard, train_lp_filtered, train_amp_filtered
from src.evaluation import test_clean, test_AA, test_gaussian
from src.args import get_args
from src.utils_dataset import load_dataset
from src.utils_freq import mask_radial
from src.utils_general import seed_everything, get_model, get_optim

def train(args, loader, model, opt, device):

    """perform one epoch of training."""
    if args.method == "standard":
        train_log = train_standard(loader, model, opt, device)

    elif args.method == "remove_high_freq":
        dim = 32

        spacing = dim*np.sqrt(2)/2000
        candidate_radius = np.arange(dim*np.sqrt(2), 0, -spacing)
        for _r in candidate_radius:
            freq_mask = mask_radial(dim, _r)
            included_freq_ratio = freq_mask.sum()/dim/dim*100
            if included_freq_ratio < args.threshold:
                break
        freq_mask = torch.tensor(freq_mask, dtype=torch.float32, device=device)
        train_log = train_lp_filtered(loader, model, opt, freq_mask, device)

    elif args.method == "remove_low_amp":
        train_log = train_amp_filtered(loader, model, opt, args.threshold, args.dataset, device)

    else:
        raise  NotImplementedError("Training method not implemented!")

    return train_log

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = get_args()
    logging.basicConfig(
        filename=args.j_dir+ "/log.txt",
        format='%(asctime)s %(message)s', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    seed_everything(args.seed)
    train_loader, test_loader = load_dataset(args.dataset, args.batch_size)

    var_list = [0.001, 0.005, 0.007]
    l2_eps_list = [0.1, 0.2, 0.3]
    linf_eps_list = [1./255., 2./255., 4./255.]

    model = get_model(args)
    model.to(device)
    if not args.eval_only:
        opt, lr_scheduler = get_optim(model.parameters(), args)
    else:
        args.epoch = 0


    for _epoch in range(0, args.epoch):
        train_log = train(args, train_loader, model, opt, device)
        if lr_scheduler:
            lr_scheduler.step()

        test_log = test_clean(test_loader, model, device)

        logging.info(
            "Epoch: [{0}]\tlr: {1:.6f}\tTrain Loss: {2:.6f}\t"
            "Train Accuracy(top1): {3:.2f}%\tTest Loss: {4:.6f}\t"
            "Test Accuracy(top1): {5:.2f}%\tTest Accuracy(top5): {6:.2f}%".format(
                _epoch, opt.param_groups[-1]['lr'], train_log[1], train_log[0],
                test_log[1], test_log[0], test_log[2]))

        if (_epoch+1) == args.epoch:
            torch.save(model.state_dict(), args.j_dir+"/final_model.pt")

    test_log = test_clean(test_loader, model, device)
    logging.info(
        "Test Accuracy(top1): {0:.2f}%\tTest Accuracy(top5): {1:.2f}%".format(
            test_log[0], test_log[2]))

    for var in var_list:
        gau_acc = test_gaussian(test_loader, model, var, device)
        logging.info("Gau(var={:.3f}): Accuracy(top1): {:.2f}%\tAccuracy(top5): {:.2f}%".format(
            var, gau_acc[0], gau_acc[1]))

    for eps in l2_eps_list:
        l2_AA_acc = test_AA(test_loader, model, norm="L2", eps=eps)
        logging.info("AA(L2, eps={:.3f}): Accuracy(top1) {:.2f}%\tAccuracy(top5): {:.2f}%".format(
            eps, l2_AA_acc[0], l2_AA_acc[1]))

    for eps in linf_eps_list:
        linf_AA_acc = test_AA(test_loader, model, norm="Linf", eps=eps)
        logging.info("AA(Linf, eps={:.3f}): Accuracy(top1) {:.2f}%\tAccuracy(top5): {:.2f}%".format(
            eps, linf_AA_acc[0], linf_AA_acc[1]))

if __name__ == "__main__":
    main()
