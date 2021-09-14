from dataloader import *
import numpy as np

import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # not use X window to show img

import time
import copy

import tqdm
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

def load_data(batch=8, preprocessing=True):    
    # train transformations
    train_trans = transforms.Compose([transforms.ToPILImage(),
                                      transforms.RandomRotation((-360, 360)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor()
                                     ])

    # validation transformations
    test_trans = transforms.Compose([transforms.ToPILImage(),
                                      transforms.ToTensor(),
                                     ])

    train = RetinopathyLoader("./data/", "train", transform=train_trans, preprocessing=preprocessing)
    test = RetinopathyLoader("./data/", "test", transform=test_trans, preprocessing=preprocessing)

    # Create training and validation dataloaders
    train_loader = DataLoader(
        dataset=train, 
        batch_size=batch,
        num_workers = 4
    )

    # [C, H, W] == [3, 512, 512]
    test_loader = DataLoader(
        dataset=test,
        batch_size=batch,
        num_workers = 4
    )
    return train_loader, test_loader


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    avg_loss, avg_acc = 0, 0
    batch = 0
    print("Training...")
    for x, y in tqdm.tqdm(dataloader):
        x = x.to(device)
        y = y.to(device)

        # Compute prediction and loss
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.item()
        avg_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
#         if batch % 1000 == 0:
#             loss, current = loss.item(), batch * len(x)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        batch+=1
    avg_loss /= len(dataloader)
    avg_acc /= len(dataloader.dataset)
    return 100*avg_acc, avg_loss


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    print("Testing...")
    with torch.no_grad():
        for X, y in tqdm.tqdm(dataloader):
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return 100*correct, test_loss

def save_acc(tr_acc, te_acc, tr_loss, te_loss):
    try:
        acc = np.load("acc.npz")
        np_train_acc = acc['train_acc']
        np_test_acc = acc['test_acc']
        np_train_loss = acc['train_loss']
        np_test_loss = acc['test_loss']
        np_train_acc = np.append(np_train_acc, tr_acc)
        np_test_acc = np.append(np_test_acc, te_acc)
        np_train_loss = np.append(np_train_loss, tr_loss)
        np_test_loss = np.append(np_test_loss, te_loss)
        np.savez('acc.npz', train_acc=np_train_acc, test_acc=np_test_acc, train_loss=np_train_loss, test_loss=np_test_loss)
    except:
        tr_acc = np.array(tr_acc)
        te_acc = np.array(te_acc)
        tr_loss = np.array(tr_loss)
        te_loss = np.array(te_loss)
        np.savez('acc.npz', train_acc=tr_acc, test_acc=te_acc, train_loss=tr_loss, test_loss=te_loss)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
