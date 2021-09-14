from dataloader import *
from utils import *
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
import argparse
import sys
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


def main():
    parser = argparse.ArgumentParser(description='Diabetic Retinopathy Detection')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    
    parser.add_argument('--train', action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument('--eval', action="store_true")
    parser.add_argument("--ResNet18", action="store_true")
    parser.add_argument("--ResNet50", action="store_true")
    parser.add_argument('--pretraining', action="store_true")
    parser.add_argument("--print_acc", action="store_true")
    parser.add_argument("--print_loss", action="store_true")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--filename", type=str, default="./model.pt")
    
    args = parser.parse_args()
    
    if not args.ResNet18 and not args.ResNet50 and not args.load_model:
        sys.exit("Not choosen the neural network. (--ResNet18 or --ResNet50)")
    
    if not args.train and not args.eval:
        sys.exit("Not choosen the mode. (--train or --eval)")

    
    print("Loading Data...")
    train_loader, test_loader = load_data(args.batch)
    print()
    
    
    feature_extract = False

    if args.ResNet18:
        print("Using ResNet18 and pretraining {}\n".format(args.pretraining))
        model = models.resnet18(pretrained=args.pretraining)
    elif args.ResNet50:
        print("Using ResNet50 and pretraining {}\n".format(args.pretraining))
        model = models.resnet50(pretrained=args.pretraining)
    elif args.load_model:
        try:
            model = torch.load(args.filename, map_location=device)
            print("Using load model: {}\n".format(args.filename))
        except:
            sys.exit("Error: The file name is not found !")
                        
    set_parameter_requires_grad(model, feature_extract)

    if args.ResNet18 or args.ResNet50:
        num_classes = 5
        num_features = model.fc.in_features # the input channel of fc layer
        model.fc = nn.Linear(num_features, num_classes)
        model = model.to(device)
        
    ## to check which params are going to learn
    print("Checking which params are going to learn... ")
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    print()
                    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params_to_update, lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

    
    if args.train:  
        train_acc = []
        train_loss = []
        test_acc = []
        test_loss = []
        for t in range(args.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            tr_acc, tr_loss = train_loop(train_loader, model, loss_fn, optimizer)
            te_acc, te_loss = test_loop(test_loader, model, loss_fn)

            if args.save:
                save_acc(tr_acc, te_acc, tr_loss, te_loss)
            else:
                train_acc.append(tr_acc)
                train_loss.append(tr_loss)
                test_acc.append(te_acc)
                test_loss.append(te_loss)


        print("Done!")

        if args.save:
            print("Saving model weight...")
            torch.save(model, "./model.pt")
            print("Save Done!")
        
        if args.print_acc:
            plt.figure(figsize=(14, 8))
            
            load = True
            if load:
                acc = np.load('acc.npz')
                train_acc = acc['train_acc']
                test_acc = acc['test_acc']

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
            plt.figure()
            
            load = True
            if load:
                acc = np.load('acc.npz')
                train_loss = acc['train_loss']
                test_loss = acc['test_loss']

            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            x = np.arange(1, len(train_loss)+1)
            plt.plot(x, train_loss, label="train")
            plt.plot(x, test_loss, label="test")
            plt.legend()
#             plt.show()
            print("Saving loss.png...")
            plt.savefig("loss.png")

    
    elif args.eval:
        model.eval()
        y_pred = []   
        y_true = []   

        test_loss, correct = 0, 0
        print("Evaluating...")
        with torch.no_grad():
            for x, y in tqdm.tqdm(test_loader):
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
            #     test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

                y_pred.extend(pred.argmax(1).detach().cpu().numpy())       # 將preds預測結果detach出來，並轉成numpy格式       
                y_true.extend(y.view(-1).detach().cpu().numpy())

        correct /= len(test_loader.dataset)

        print(f"Test Error: \n Accuracy: {(100*correct)}%")
        
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        annot = np.around(cm, 2)

        # plot matrix
        fig, ax = plt.subplots(figsize = (12, 10))
        sns.heatmap(cm, cmap = 'Blues', annot = annot, lw = 0.5)
        ax.set_xlabel('Prediction', fontsize=12)
        ax.set_ylabel('Ground Truth', fontsize=12)
        ax.set_aspect('equal')
        plt.savefig("confusion_matrix.png")
        
    

if __name__ == '__main__':
    main()