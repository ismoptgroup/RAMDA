import torch
import torchvision

import argparse
import os

from model import ConvNet

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./')
args = parser.parse_args()

if not os.path.exists(args.path+'Models'):
    os.makedirs(args.path+'Models')
    
    convnet_fashionmnist_1 = ConvNet()
    torch.save(convnet_fashionmnist_1.state_dict(), args.path+'Models/ConvNet_FashionMNIST_1.pt')

torchvision.datasets.FashionMNIST(root=args.path+'Data', train=True, download=True)
torchvision.datasets.FashionMNIST(root=args.path+'Data', train=False, download=True)
    
if not os.path.exists(args.path+'Saved_Models'):
    os.makedirs(args.path+'Saved_Models')

if not os.path.exists(args.path+'Results'):
    os.makedirs(args.path+'Results')
    if not os.path.exists(args.path+'Results/Presentation'):
        os.makedirs(args.path+'Results/Presentation')
    if not os.path.exists(args.path+'Results/ForPlotting'):
        os.makedirs(args.path+'Results/ForPlotting')