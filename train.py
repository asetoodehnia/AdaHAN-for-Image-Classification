import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


import pandas as pd

import matplotlib.pyplot as plt
import numpy as np


def train(model, criterion, optimizer, train_set, val_set, num_epochs, batch_size, device, attn=True):
    model.to(device)
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    val_loader = torch.utils.data.DataLoader(val_set, 
                                             batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    size = int(len(train_set) / batch_size)

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        for i, (_, inputs, labels) in enumerate(train_loader):
            # get the inputs and labels
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if attn:
                class_pred, latent_mask = model(inputs)
            else:
                class_pred = model(inputs)
            loss = criterion(class_pred, labels)
            loss.backward()
            optimizer.step()

            # calculate running loss and acc
            running_loss += loss.item()
            _, predicted = torch.max(class_pred.data, 1)
            running_acc += (predicted == labels).sum().item() / labels.size(0)
            
            # print every (size / 5) mini-batches
            if i % (size / 5) == (size / 5) - 1:
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / (size / 5)))
                train_losses.append(running_loss / (size / 5))
                train_accs.append(running_acc / (size / 5))
                running_loss = 0.0
                running_acc = 0.0

        val_loss, val_acc = get_loss_and_acc(model, criterion, 
                                             val_loader, device, attn)
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
    print('Finished Training')
    return train_losses, train_accs, val_losses, val_accs


def get_loss_and_acc(model, criterion, data_loader, device, attn=True):
    correct = 0
    total = 0
    running_loss = 0
    i = 0
    model.eval()
    with torch.no_grad():
        for i, (_, inputs, labels) in enumerate(data_loader):
            # get the inputs and labels
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # forward
            if attn:
                class_pred, latent_mask = model(inputs)
            else:
                class_pred = model(inputs)
            
            # loss
            loss = criterion(class_pred, labels)
            running_loss += loss.item()
            
            # accuracy
            _, predicted = torch.max(class_pred.data, 1)
            total += labels.size(0)
            i += 1
            correct += (predicted == labels).sum().item()

    print('Accuracy: %f ' % (correct / total))
    print('Loss: %f ' % (running_loss / i))

    return running_loss / i, correct / total


def get_test_acc(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (_, inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            class_pred, latent_mask = model(inputs)
            _, predicted = torch.max(class_pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %f' % (correct / total))


def get_test_class_acc(model, classes, test_loader, device):
    model.eval()
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for i, (_, inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            class_pred, latent_mask = model(inputs)
            _, predicted = torch.max(class_pred, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(classes)):
        print('Accuracy of %5s : %2f' % (classes[i], class_correct[i] / class_total[i]))
