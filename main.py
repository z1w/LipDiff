import torch
from torch import nn
import numpy as np
from model import Mnist_model, Mnist_model3, Mnist_model_deep, Mnist_CNN, NeuralNet, Mnist_c2f, Mnist_CNN_large, Mnist_c2f_large, cifar_tiny, cifar_large
from data import MNIST_DATA, CIFAR10_DATA
from SDP_solver import LipSDP, EigSDP, ReduntSDP
import math
import os
from numpy import linalg as LA
from FO_solver import FOSDP
from utils import norm_prod, extract_network, test_model
import time

import argparse


def create_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", 
        nargs='?', 
        const="dnn", 
        default="dnn", 
        choices=['dnn', 'cnn'], 
        help="which model to use")

    parser.add_argument("--device", 
        type=int,
        default=2,
        help="which gpu to use")
    
    parser.add_argument("--batch_size", 
        type=int,
        default=128,
        help="how large is a batch during training")
    
    parser.add_argument("--epochs", 
        type=int,
        default=200,
        help="how many epochs for training")
    
    parser.add_argument("--data", 
        nargs='?', 
        const="mnist", 
        default="mnist", 
        choices=['mnist', 'cifar10'], 
        help="which data to use")

    
    parser.add_argument("--lr", 
        type=float,
        default=0.02, 
        help="learning rate")
    
    parser.add_argument("--momentum", 
        type=float,
        default=0.9, 
        help="momemtum")
    
    parser.add_argument("--decay", 
        type=float,
        default=8e-4, 
        help="weight decay")
    
    parser.add_argument("--method", 
        nargs='?', 
        const="product", 
        default="product", 
        choices=['product', 'lipsdp', 'fo_solver'], 
        help="which method to use")

    parser.add_argument("--fo_iters", 
        type=int,
        default=600,
        help="how many iterations for first order method")
    
    parser.add_argument("--fo_lr", 
        type=float,
        default=0.03, 
        help="learning rate")
    
    parser.add_argument("--lan_steps", 
        type=int,
        default=30,
        help="how many lanczos iterations are used")
    
    parser.add_argument("--verbose", 
        action="store_true",
        help="increase output verbosity")

    parser.add_argument("--lanczos", 
        action="store_true",
        help="using lanczos algorithm to estimate the eigenvalue")
    
    parser.add_argument("--sparse", 
        action="store_true",
        help="sparse representation of the constraint matrix for multiplication")
    
    parser.add_argument("--init", 
        nargs='?', 
        const="random", 
        default="random", 
        choices=['random', 'schur'], 
        help="how to initialize the variables")
    
    
    parser.add_argument("--module", 
        action="store_true",
        help="using torch native module multiplication instead of the matrix representation")
    
    parser.add_argument("--groups", 
        type=int,
        default=2,
        help="how many groups for optimization")

    return parser

def main(args):
    if not os.path.exists('models'):
        os.makedirs('models')
    
    device = "cuda:"+str(args.device)
    if args.data == "mnist":
        data = MNIST_DATA(args.batch_size, 1000)
        if args.model == "dnn":
            path = "models/mnist_dnn.pth"
            model = Mnist_model(args.epochs, path, device).to(device)
        else:
            path = "models/mnist_cnn.pth"
            model = Mnist_CNN(args.epochs, path, device).to(device)
        img_size = [28, 28]
        channels = 1
    else:
        data = CIFAR10_DATA(args.batch_size, 1000)
        path = "models/cifar10_cnn.pth"
        model = cifar_tiny(args.epochs, path, device).to(device)
        img_size = [32, 32]
        channels = 3
    
    if not os.path.exists(path):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
        model.train_model(data.train_loader, optimizer, criterion)
        model.save_model()
    else:
        print("Load trained model...")
        model.load_model()
    test_model(model, data)
    weights, weight_types, params = extract_network(model, in_size=img_size, channels=channels)
    label = [8]

    seconds = time.time()
    if args.method == "product":
        print("The norm product is:", norm_prod(weights, label), ", which takes time", time.time()-seconds)
    elif args.method == "lipsdp":
        solver = LipSDP(weights, label)
        print("LipSDP solver gives:", solver.solve_sdp(verbose=args.verbose), ", which takes time", time.time()-seconds)
    else:
        solver = FOSDP(weights, pair=label, weight_types=weight_types, params=params, epochs=args.fo_iters, device=device, memory_eff=False)
        print("The FO SDP gives:", solver.solve_eig(lr=args.fo_lr, lanc=args.lanczos, sparse=args.sparse, init=args.init, verbose=args.verbose, lan_steps=args.lan_steps, ratio=10, module=args.module, group_val=args.groups), ", which takes time", time.time()-seconds)


    



if __name__ == '__main__':
    parser = create_args()
    args = parser.parse_args()

    main(args)