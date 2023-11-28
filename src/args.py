import yaml
import argparse
import os
from src.utils_general import DictWrapper
import distutils.util
    
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method",
                        default=argparse.SUPPRESS)
    parser.add_argument("--dataset",
                        default=argparse.SUPPRESS)
    parser.add_argument("--arch",
                        default=argparse.SUPPRESS)
    parser.add_argument("--pretrain",
                        default=None, type=str)

    # hyper-param for optimization
    parser.add_argument("--optim",
    			default=argparse.SUPPRESS)
    parser.add_argument("--lr",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--lr_scheduler_type",
    			default=argparse.SUPPRESS)
    parser.add_argument("--momentum",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--weight_decay",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--adam_beta1",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--adam_beta2",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--rmsp_alpha",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--batch_size",
    			default=argparse.SUPPRESS, type=int)
    parser.add_argument("--seed",
    			default=argparse.SUPPRESS, type=int)
    parser.add_argument("--epoch",
    			default=argparse.SUPPRESS, type=int)

    parser.add_argument("--j_dir",
    			default='./exp')

    parser.add_argument("--lr_warmup_type",
    			default=argparse.SUPPRESS)
    parser.add_argument("--lr_warmup_epoch",
    			default=argparse.SUPPRESS, type=int)
    parser.add_argument("--lr_warmup_decay",
    			default=argparse.SUPPRESS, type=float)

    parser.add_argument("--threshold",
    			default=90, type=float)

    parser.add_argument('--input_normalization',
                        default=False, type=distutils.util.strtobool)
    parser.add_argument('--enable_batchnorm',
                        default=False, type=distutils.util.strtobool)
    
    parser.add_argument('--eval_only',
                        action="store_true")


    args = parser.parse_args()

    return args

def get_default(yaml_path):
    default = {}
    with open(yaml_path, 'r') as handle:
        default = yaml.load(handle, Loader=yaml.FullLoader)
    return default 

def get_args():
    args = parse_args()
    default = get_default('options/default.yaml')
    
    default.update(vars(args).items())
    args_dict = DictWrapper(default)

    return args_dict

