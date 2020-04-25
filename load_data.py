import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import datasets

import matplotlib.pyplot as plt
import numpy as np


def get_cifar10_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainset, trainloader, testset, testloader, classes

def get_miniplaces_data():
    trainset = datasets.MiniplacesDataset('train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)

    testset = datasets.MiniplacesDataset('test')
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=2)

    classes = []
    txt_root = './miniplaces/development_kit/data'
    txt_dir = os.path.join(txt_root, "categories" + '.txt')
    with open(txt_dir, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            classes.append(tokens[0])
                
    return trainset, trainloader, testset, testloader, classes
