# LipDiff
This repo contains the code for an ICLR submission, a first-order SDP solver for the Lipschitz constant estimation of deep neural networks.

## Prerequisites
You need Torch packages (torch and torchvision), cvxpy and mosek solver for this project.

## Instructions
We have three data/model combos: MNIST DNN, an MNIST CNN (trained for 50 epochs, accuracy > 97%) and a CIFAR10 CNN (trained for 200 epochs, accuracy > 80%). When specifying the data and model combos, the script will check out whether these models have been trained. If no, the script will train the model and then save it.

We have three methods to estimate the Lipschitz constant (of label 8): matrix norm product, LipSDP, and the first-order method.

`python3 main.py --data mnist --model dnn --epochs 50 --batch_size 256 --device 1 --method product` 

A sample output might be:
> Load trained model...
>
> accuracy is 97.89%
>
> Number of parameters: 203530
>
> The norm product is: 9.308638302016163 , which takes time 0.03991222381591797

`python3 main.py --data mnist --model dnn --epochs 50 --batch_size 256 --device 1 --method lipsdp`

A sample output can be:
> Load trained model...
> 
> accuracy is 97.89%
>
> Number of parameters: 203530
> 
> LipSDP solver gives: 4.821720098724499 , which takes time 46.962613582611084

`python3 main.py --data mnist --model dnn --epochs 50 --batch_size 256 --device 0 --method fo_solver --lanczos --init schur --fo_lr 0.04 --lan_steps 20 --groups 3`

>The FO SDP gives: tensor(4.9786, device='cuda:0', dtype=torch.float64, grad_fn=<DivBackward0>) , which takes time 9.847759008407593

## Citation
If you want to cite this work, you might use the following bibtex:
```
@inproceedings{
wang2024on,
title={On the Scalability and Memory Efficiency of Semidefinite Programs  for Lipschitz Constant Estimation of Neural Networks},
author={Zi Wang and Bin Hu and Aaron J Havens and Alexandre Araujo and Yang Zheng and Yudong Chen and Somesh Jha},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=dwzLn78jq7}
}
```



