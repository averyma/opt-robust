# Understanding the robustness difference between stochastic gradient descent and adaptive gradient methods

This is the official repository of "Understanding the robustness difference between stochastic gradient descent and adaptive gradient methods" accepted at TMLR with Featured Certification.

## Requirements
This code has been tested with  
python 3.9.15   
pytorch 2.0.1   
torchvision 0.15.2   
numpy 1.22.4   

## Checkpoints
We provide six model [checkpoints](https://drive.google.com/drive/folders/1es5dmvHL35uPfUHclxvylA2dS_LNlS33?usp=drive_link). They are PreActResNet18 models trained on CIFAR10/100 with SGD, Adam, and RMSProp. 

| Model | Standard Accuracy | Gaussian ($\sigma^2 = 0.007$) | $\ell_2$ AutoAttack ($\epsilon = 0.3$) | $\ell_\infty$ AutoAttack ($\epsilon = \frac{4}{255}$) |
| -------- | -------- | -------- | -------- | -------- |
| cifar10-sgd      | 89.86%   | 69.54%   | 17.50%    | 1.00%    | 
| cifar10-adam     | 90.77%   | 58.70%   | 11.10%    | 0.20%    |
| cifar10-rmsprop  | 90.64%   | 62.93%   | 8.90%     | 0.50%    |
| cifar100-sgd     | 83.64%   | 65.46%   | 57.40%    | 39.00%   |
| cifar100-adam    | 86.70%   | 49.14%   | 52.20%    | 33.60%   |
| cifar100-rmsprop | 86.08%   | 52.13%   | 56.50%    | 36.60%   |

Additionally, we provide the weight adaptation history for the training of the linear model on the synthetic dataset.


## Reproducing results
- To evaluate model robustness on Gaussian noise, $\ell_2$ and $\ell_\infty$ AutoAttacks with various $\epsilon$, run:
```
python3 main.py --dataset cifar10 --pretrain './ckpt/cifar10-adam.pt' --j_dir './exp' --eval_only
```

- Standard training on PreActResNet18 with SGD 200 epochs**, run:
```
python3 main.py --method standard --dataset cifar10 --j_dir './exp' --optim sgd --epoch 200 --lr 0.01 
```

- Training with augmented data by removing high-frequency component, run:
```
python3 main.py --method remove_high_freq --threshold 90 --dataset cifar10 --j_dir './exp' --optim sgd --epoch 200 --lr 0.01
```

- Training with augmented data by removing low spectral energy component, run:
```
python3 main.py --method remove_low_amp --threshold 90 --dataset cifar10 --j_dir './exp' --optim sgd --epoch 200 --lr 0.01
```

## Observing the presence of irrelevant frequency in natural datasets
1. Visualizing the spectral energy of natural dataset.
2. Visualizing augmented images (removing components with high frequency or low spectral energy).
3. Evaluating model robustness under band-limited Gaussian noise.

## Training linear models on the synthetic dataset and plotting the results
1.
2.

## Citing this Work 
```
@article{
ma2023understanding,
title={Understanding the robustness difference between stochastic gradient descent and adaptive gradient methods
},
author={Avery Ma and Yangchen Pan and Amir-massoud Farahmand},
journal={Transactions on Machine Learning Research},
year={2023},
url={https://openreview.net/forum?id=ed8SkMdYFT},
note={Featured Certification}
}
```

## License
MIT License

