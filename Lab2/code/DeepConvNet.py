import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepConvNet_relu(nn.Module):
    def __init__(self):
        super(DeepConvNet_relu, self).__init__()
        self.T = 750
        self.C = 2
        self.classes = 2
        self.dropout_rate = 0.5
        
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(1, 5))
        
        self.conv2 = nn.Conv2d(25, 25, kernel_size=(self.C, 1))
        self.batchnorm1 = nn.BatchNorm2d(25)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout1 = nn.Dropout(self.dropout_rate)
        
        self.conv3 = nn.Conv2d(25, 50, kernel_size=(1, 5))
        self.batchnorm2 = nn.BatchNorm2d(50)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout2 = nn.Dropout(self.dropout_rate)
        
        self.conv4 = nn.Conv2d(50, 100, kernel_size=(1, 5))
        self.batchnorm3 = nn.BatchNorm2d(100)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout3 = nn.Dropout(self.dropout_rate)
        
        self.conv5 = nn.Conv2d(100, 200, kernel_size=(1, 5))
        self.batchnorm4 = nn.BatchNorm2d(200)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout4 = nn.Dropout(self.dropout_rate)
        
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(8600, self.classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        
        x = self.conv3(x)
        x = self.batchnorm2(x)
        x - self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
        
        x = self.conv5(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.dropout4(x)
        
        x = self.flatten(x)
        x = self.dense(x)
        
        return x
    
class DeepConvNet_lrelu(nn.Module):
    def __init__(self):
        super(DeepConvNet_lrelu, self).__init__()
        self.T = 750
        self.C = 2
        self.classes = 2
        self.dropout_rate = 0.5
        
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(1, 5))
        
        self.conv2 = nn.Conv2d(25, 25, kernel_size=(self.C, 1))
        self.batchnorm1 = nn.BatchNorm2d(25)
        self.leakyrelu1 = nn.LeakyReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout1 = nn.Dropout(self.dropout_rate)
        
        self.conv3 = nn.Conv2d(25, 50, kernel_size=(1, 5))
        self.batchnorm2 = nn.BatchNorm2d(50)
        self.leakyrelu2 = nn.LeakyReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout2 = nn.Dropout(self.dropout_rate)
        
        self.conv4 = nn.Conv2d(50, 100, kernel_size=(1, 5))
        self.batchnorm3 = nn.BatchNorm2d(100)
        self.leakyrelu3 = nn.LeakyReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout3 = nn.Dropout(self.dropout_rate)
        
        self.conv5 = nn.Conv2d(100, 200, kernel_size=(1, 5))
        self.batchnorm4 = nn.BatchNorm2d(200)
        self.leakyrelu4 = nn.LeakyReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout4 = nn.Dropout(self.dropout_rate)
        
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(8600, self.classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = self.leakyrelu1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        
        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = self.leakyrelu2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = self.leakyrelu3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
        
        x = self.conv5(x)
        x = self.batchnorm4(x)
        x = self.leakyrelu4(x)
        x = self.maxpool4(x)
        x = self.dropout4(x)
        
        x = self.flatten(x)
        x = self.dense(x)
        
        return x
    
class DeepConvNet_elu(nn.Module):
    def __init__(self):
        super(DeepConvNet_elu, self).__init__()
        self.T = 750
        self.C = 2
        self.classes = 2
        self.dropout_rate = 0.5
        
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(1, 5))
        
        self.conv2 = nn.Conv2d(25, 25, kernel_size=(self.C, 1))
        self.batchnorm1 = nn.BatchNorm2d(25)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout1 = nn.Dropout(self.dropout_rate)
        
        self.conv3 = nn.Conv2d(25, 50, kernel_size=(1, 5))
        self.batchnorm2 = nn.BatchNorm2d(50)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout2 = nn.Dropout(self.dropout_rate)
        
        self.conv4 = nn.Conv2d(50, 100, kernel_size=(1, 5))
        self.batchnorm3 = nn.BatchNorm2d(100)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout3 = nn.Dropout(self.dropout_rate)
        
        self.conv5 = nn.Conv2d(100, 200, kernel_size=(1, 5))
        self.batchnorm4 = nn.BatchNorm2d(200)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout4 = nn.Dropout(self.dropout_rate)
        
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(8600, self.classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        
        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
        
        x = self.conv5(x)
        x = self.batchnorm4(x)
        x = F.elu(x)
        x = self.maxpool4(x)
        x = self.dropout4(x)
        
        x = self.flatten(x)
        x = self.dense(x)
        
        return x