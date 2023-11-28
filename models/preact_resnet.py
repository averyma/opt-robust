'''https://raw.githubusercontent.com/kuangliu/pytorch-cifar/master/models/preact_resnet.py'''

'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, enable_batchnorm=True):
        super(PreActBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.enable_batchnorm = enable_batchnorm
        if enable_batchnorm:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(planes)


        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x)) if self.enable_batchnorm else F.relu(x)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out))) if self.enable_batchnorm else self.conv2(F.relu(out))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, enable_batchnorm=True):
        super(PreActBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.enable_batchnorm = enable_batchnorm
        if enable_batchnorm:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bn3 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x)) if self.enable_batchnorm else F.relu(x)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out))) if self.enable_batchnorm else self.conv2(F.relu(out))
        out = self.conv3(F.relu(self.bn3(out))) if self.enable_batchnorm else self.conv3(F.relu(out))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, dataset='cifar10', num_classes=10, input_normalization=True, enable_batchnorm=True):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        if dataset in ['cifar10', 'svhn', 'cifar100']:
#         if dataset in ['cifar10', 'svhn', 'cifar100', 'dtd']:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # elif dataset in ['tiny', 'dtd']:
        elif dataset in ['tiny', ]:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
#         elif dataset in ['imagenette']:
        elif dataset in ['imagenette', 'dtd','caltech','fmd','flowers','cars','food']:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, enable_batchnorm=enable_batchnorm)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, enable_batchnorm=enable_batchnorm)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, enable_batchnorm=enable_batchnorm)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, enable_batchnorm=enable_batchnorm)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.input_normalization = input_normalization
        self.dataset = dataset

    def _make_layer(self, block, planes, num_blocks, stride, enable_batchnorm):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, enable_batchnorm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def per_image_standardization(self, x):
        """
        https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
        """
        _dim = x.shape[1] * x.shape[2] * x.shape[3]
        mean = torch.mean(x, dim=(1,2,3), keepdim = True)
        stddev = torch.std(x, dim=(1,2,3), keepdim = True)
        adjusted_stddev = torch.max(stddev, (1./np.sqrt(_dim)) * torch.ones_like(stddev))
        return (x - mean) / adjusted_stddev

    def forward(self, x):
        if self.input_normalization:
            x = self.per_image_standardization(x)
        out = self.conv1(x)
        # if self.dataset in ['imagenette', 'imagenet']:
        # if self.dataset in ['imagenette', 'imagenet','dtd','caltech','fmd','flowers']:
        if self.dataset in ['imagenette', 'dtd','caltech','fmd','flowers','cars','food']:
            out = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # if self.dataset in ['imagenette', 'imagenet']:
        # if self.dataset in ['imagenette', 'imagenet','dtd','caltech','fmd','flowers']:
        if self.dataset in ['imagenette', 'dtd','caltech','fmd','flowers','cars','food']:
            out = F.avg_pool2d(out, 7)
        else:
            out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def return_z(self, x):
        if self.input_normalization:
            x = self.per_image_standardization(x)
        out = self.conv1(x)
        # if self.dataset in ['imagenette', 'imagenet']:
        # if self.dataset in ['imagenette', 'imagenet','dtd','caltech','fmd','flowers']:
        if self.dataset in ['imagenette', 'dtd','caltech','fmd','flowers','cars','food']:
            out = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # if self.dataset in ['imagenette', 'imagenet']:
        # if self.dataset in ['imagenette', 'imagenet','dtd','caltech','fmd','flowers']:
        if self.dataset in ['imagenette', 'dtd','caltech','fmd','flowers','cars','food']:
            out = F.avg_pool2d(out, 7)
        else:
            out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def PreActResNet18(dataset, num_classes, input_normalization, enable_batchnorm):
    return PreActResNet(PreActBlock, [2,2,2,2], dataset, num_classes, input_normalization, enable_batchnorm)


def test():
    net = PreActResNet18(10,True)
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

# test()
