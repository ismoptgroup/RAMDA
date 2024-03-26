import torch
import torchvision

import argparse
import os

from model import Linear

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./')
args = parser.parse_args()

if not os.path.exists(args.path+'Models'):
    os.makedirs(args.path+'Models')
    
    linear_mnist_1 = Linear()
    torch.save(linear_mnist_1.state_dict(), args.path+'Models/Linear_MNIST_1.pt')

torchvision.datasets.MNIST(root=args.path+'Data', train=True, download=True)
torchvision.datasets.MNIST(root=args.path+'Data', train=False, download=True)
    
if not os.path.exists(args.path+'Saved_Models'):
    os.makedirs(args.path+'Saved_Models')

if not os.path.exists(args.path+'Results'):
    os.makedirs(args.path+'Results')
    if not os.path.exists(args.path+'Results/Presentation'):
        os.makedirs(args.path+'Results/Presentation')
    if not os.path.exists(args.path+'Results/ForPlotting'):
        os.makedirs(args.path+'Results/ForPlotting')
