# Understanding the robustness difference between stochastic gradient descent and adaptive gradient methods

This is the official repository of "Understanding the robustness difference between stochastic gradient descent and adaptive gradient methods" accepted at Transactions on Machine Learning Research (TMLR).

![Comparing model robustness](figures/comparison.png)
**Comparison between models trained using SGD, Adam, and RMSProp across seven benchmark datasets.** 
Each colored triplet denotes models on the same dataset. Models trained by different algorithms have similar standard generalization performance, but there is a distinct robustness difference as measured by the test data accuracy under Gaussian noise and adversarial perturbations. 

## Requirements
To run the code, the following packages are needed:
- Python 3.9.15
- PyTorch 2.0.1
- torchvision 0.15.2
- numpy 1.22.4

## Checkpoints
Access our model checkpoints [here](https://drive.google.com/drive/folders/1es5dmvHL35uPfUHclxvylA2dS_LNlS33?usp=drive_link), including PreActResNet18 models trained on CIFAR10/100 using SGD, Adam, and RMSProp. For detailed settings, refer to our paper.

## Model training and evaluation
- To evaluate model robustness under Gaussian noise, $\ell_2$ and $\ell_\infty$ AutoAttacks with various $\epsilon$, run:
```
python3 main.py --eval_only --dataset cifar10 --pretrain './ckpt/cifar10-adam.pt' --j_dir './exp'
```

- To perform standard training on PreActResNet18 with SGD for 200 epochs:
```
python3 main.py --method standard --dataset cifar10 --j_dir './exp' --optim sgd --epoch 200 --lr 0.2 --lr_scheduler_type multistep --weight_decay 0
```

- To train with augmented data by removing parts of the signal with low spectrum energy:
```
python3 main.py --method remove_low_amp --threshold 90 --dataset cifar10 --j_dir './exp' --optim sgd --epoch 200 --lr 0.2 --lr_scheduler_type multistep --weight_decay 0
```

- To train with augmented data by removing parts of the signal with high frequencies:
```
python3 main.py --method remove_high_freq --threshold 90 --dataset cifar10 --j_dir './exp' --optim sgd --epoch 200 --lr 0.2 --lr_scheduler_type multistep --weight_decay 0
```
## Observing the Presence of Irrelevant Frequencies in Natural Datasets
1. **Spectral Energy Visualization**: Explore the spectral energy distribution of natural datasets. [notebook](./notebook/fig8_spectral_energy.ipynb)
2. **Augmented Image Visualization**: See how removing high-frequency or low spectral energy components affects images. [notebook](./notebook/fig12_augmented_images.ipynb)
3. **Model Robustness Evaluation**: Assess model robustness under band-limited Gaussian noises. [notebook](./notebook/fig4_band_limited_gaussian.ipynb)

## Linear Regression Analysis with an Over-parameterized Model
1. **Training Linear Models**: Train linear models on a three-dimensional synthetic dataset using GD, signGD, Adam, and RMSProp. [notebook](./notebook/train_linear_model.ipynb)
2. **Error Dynamics and Risks Plotting**: Visualize the dynamics of the error term, standard population risk, and adversarial population risk. [notebook](./notebook/fig5_linear_analysis.ipynb)


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
