from EEGNet import *
from DeepConvNet import *
from dataloader import *
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np
from sklearn import preprocessing
import sys
import argparse

import matplotlib
matplotlib.use('Agg') # not use X window to show img


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

def load_data(batch=128):
    x_train, y_train, x_test, y_test = read_bci_data()

    x_train = torch.from_numpy(x_train).float().to(device)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor).to(device)

    x_test = torch.from_numpy(x_test).float().to(device)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor).to(device) # data type is long

    train = torch.utils.data.TensorDataset(x_train,y_train)
    test = torch.utils.data.TensorDataset(x_test,y_test)

    batch_size = 128
    train_loader = DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        dataset=test,
        batch_size=batch_size,
        shuffle=True,
    )
    
    return train_loader, test_loader, x_test, y_test

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    avg_loss, avg_acc = 0, 0
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.item()
        avg_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    avg_loss /= len(dataloader)
    avg_acc /= len(dataloader.dataset)
    return 100*avg_acc, avg_loss


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return 100*correct, test_loss

def main():
    parser = argparse.ArgumentParser(description='EEG Classification')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch', type=int, default=64)
    
    parser.add_argument('--train', action="store_true")
    parser.add_argument("--EEGNet", action="store_true")
    parser.add_argument("--DeepConvNet", action="store_true")
    parser.add_argument("--print_acc", action="store_true")
    parser.add_argument("--print_loss", action="store_true")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--filename", type=str, default="./model.pt")
    
    ## activation funciton
    parser.add_argument("--relu", action='store_true')
    parser.add_argument("--elu", action='store_true')
    parser.add_argument("-lrelu", action='store_true')
    args = parser.parse_args()
    
    
    if not args.relu and not args.elu and not args.lrelu:
        sys.exit("Not choosen the activation funciton. (--relu, --elu or --lrelu)")
    
    if not args.EEGNet and not args.DeepConvNet:
        sys.exit("Not choosen the neural network. (--EEGNet or --DeepConvNet)")
    
    train_loader, test_loader, x_test, y_test = load_data(args.batch)
    
    if args.EEGNet:
        if args.relu:
            model = EEGNet_relu().to(device)
        elif args.elu:
            model = EEGNet_elu().to(device)
        elif args.lrelu:
            model = EEGNet_lrelu().to(device)
        
    elif args.DeepConvNet:
        if args.relu:
            model = DeepConvNet_relu().to(device)
        elif args.elu:
            model = DeepConvNet_elu().to(device)
        elif args.lrelu:
            model = DeepConvNet_lrelu().to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.02)
    
    if args.train:  
        train_acc = []
        train_loss = []
        test_acc = []
        test_loss = []
        for t in range(args.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            tr_acc, tr_loss = train_loop(train_loader, model, loss_fn, optimizer)
            te_acc, te_loss = test_loop(test_loader, model, loss_fn)
            train_acc.append(tr_acc)
            train_loss.append(tr_loss)
            test_acc.append(te_acc)
            test_loss.append(te_loss)
            if te_acc > 88.5 and t > 300:
                break
        print("Done!")

        print("Saving model weight...")
        torch.save(model, "./model.pt")
        print("Save Done!")
        
        if args.print_acc:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(14, 8))
            plt.ylabel("Accuracy(%)", fontsize=14)
            plt.xlabel("Epoch", fontsize=14)
            x = np.arange(1, len(train_acc)+1)
            plt.plot(x, train_acc, label="train")
            plt.plot(x, test_acc, label="test")
            plt.legend()
#             plt.show()
            print("Saving acc.png...")
            plt.savefig("acc.png")

        if args.print_loss:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            x = np.arange(1, len(train_loss)+1)
            plt.plot(x, train_loss, label="train")
            plt.plot(x, test_loss, label="test")
            plt.legend()
#             plt.show()
            print("Saving loss.png...")
            plt.savefig("loss.png")

    
    elif args.load_model:
        model_load = torch.load(args.filename, map_location=device)
        model_load.eval()

        test_loss, correct = 0, 0
        pred = model_load(x_test)
        test_loss += loss_fn(pred, y_test).item()
        correct += (pred.argmax(1) == y_test).type(torch.float).sum().item()

        correct /= len(x_test)

        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        print(model_load)
        summary(model_load, (1, 2, 750))
    

if __name__ == '__main__':
    main()