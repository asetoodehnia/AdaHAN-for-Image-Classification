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

def get_miniplaces_data(batch_size):
    train_set = datasets.MiniplacesDataset('train')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    val_set = datasets.MiniplacesDataset('val')
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)

    test_set = datasets.MiniplacesDataset('test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=2)

    classes = []
    txt_root = './miniplaces/development_kit/data'
    txt_dir = os.path.join(txt_root, "categories" + '.txt')
    with open(txt_dir, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            classes.append(tokens[0])
                
    return train_set, train_loader, val_set, val_loader, test_set, test_loader, classes
