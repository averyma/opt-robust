"""
AM: Aug28, 2019 rename model l1, l2 and l3 as linear1, linear2 and linear 3
    edited main.py to reflect change, for binary_mnist_l2*, i changed their 
    name to binary_mnist_linear2 in both the actual weight file and the record
    in the log.txt file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb

class linear_model(nn.Module):
    """
    Typically, during initialization, weights are sampled from 
    1. Normal distribution with zero mean and std^2 where std is computed using 1/sqrt(features)
        std is inversely proportional to the dim
        with large nn, we can argue that std is very small, so weights are iniatilized around 0
    2. Uniform{-k, k}, where k = 1/sqrt(input features)
    """
    def __init__(self, dim):
        super(linear_model, self).__init__()
        self.linear = nn.Linear(dim, 1, bias = False)
        
    def forward(self, x):
        output = self.linear(x.t())
        return output

