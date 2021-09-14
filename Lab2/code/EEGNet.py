import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseConv2d(torch.nn.Conv2d):
    def __init__(self,
                 in_channels,
                 depth_multiplier=1,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros'
                 ):
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode
        )
        
class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, bias=bias, padding=padding)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias)
    
    def forward(self,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class EEGNet_relu(nn.Module):
    def __init__(self):
        super(EEGNet_relu, self).__init__()
        self.C = 2
        self.T = 750
        self.F1 = 16
        self.D = 2
        self.F2 = 32
        self.kernel_length = 64
        self.dropoutRate = 0.2
        self.classes = 2
        
        # block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(1, 51), padding=(0, 25), stride=(1, 1), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(self.F1)
        self.depthwise1 = DepthwiseConv2d(self.F1, kernel_size=(self.C, 1), depth_multiplier=self.D, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(self.F1*self.D)
        self.relu1 = nn.ReLU()
        self.averagePooling1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(p=self.dropoutRate)
        
        # block 2
        self.separableconv1 = SeparableConv2d(self.F1*self.D, self.F2, kernel_size=(1, 15), padding=(0, 7), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(self.F2)
        self.relu2 = nn.ReLU()
        self.averagePooling2 = nn.AvgPool2d(kernel_size=(1,8))
        self.dropout2 = nn.Dropout(p=self.dropoutRate)
        self.flatten1 = nn.Flatten()
        
        self.dense1 = nn.Linear(self.F2*(self.T//32), self.classes)
        
        
    def forward(self, x):
        
        ## block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise1(x)
        x = self.batchnorm2(x)
        x = self.relu1(x)
        x = self.averagePooling1(x)
        x = self.dropout1(x)
        
        ## block 2
        x = self.separableconv1(x)
        x = self.batchnorm3(x)
        x = self.relu2(x)
        x = self.averagePooling2(x)
        x = self.dropout2(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        
        return x
    
class EEGNet_lrelu(nn.Module):
    def __init__(self):
        super(EEGNet_lrelu, self).__init__()
        self.C = 2
        self.T = 750
        self.F1 = 16
        self.D = 2
        self.F2 = 32
        self.kernel_length = 64
        self.dropoutRate = 0.2
        self.classes = 2
        
        # block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(1, 51), padding=(0, 25), stride=(1, 1), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(self.F1)
        self.depthwise1 = DepthwiseConv2d(self.F1, kernel_size=(self.C, 1), depth_multiplier=self.D, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(self.F1*self.D)
        self.lrelu1 = nn.LeakyReLU()
        self.averagePooling1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(p=self.dropoutRate)
        
        # block 2
        self.separableconv1 = SeparableConv2d(self.F1*self.D, self.F2, kernel_size=(1, 15), padding=(0, 7), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(self.F2)
        self.lrelu2 = nn.LeakyReLU()
        self.averagePooling2 = nn.AvgPool2d(kernel_size=(1,8))
        self.dropout2 = nn.Dropout(p=self.dropoutRate)
        self.flatten1 = nn.Flatten()
        
        self.dense1 = nn.Linear(self.F2*(self.T//32), self.classes)
        
        
    def forward(self, x):
        
        ## block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise1(x)
        x = self.batchnorm2(x)
        x = self.lrelu1(x)
        x = self.averagePooling1(x)
        x = self.dropout1(x)
        
        ## block 2
        x = self.separableconv1(x)
        x = self.batchnorm3(x)
        x = self.lrelu2(x)
        x = self.averagePooling2(x)
        x = self.dropout2(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        
        return x
    
class EEGNet_elu(nn.Module):
    def __init__(self):
        super(EEGNet_elu, self).__init__()
        self.C = 2
        self.T = 750
        self.F1 = 16
        self.D = 2
        self.F2 = 32
        self.kernel_length = 64
        self.dropoutRate = 0.2
        self.classes = 2
        
        # block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(1, 51), padding=(0, 25), stride=(1, 1), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(self.F1)
        self.depthwise1 = DepthwiseConv2d(self.F1, kernel_size=(self.C, 1), depth_multiplier=self.D, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(self.F1*self.D)
        self.averagePooling1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(p=self.dropoutRate)
        
        # block 2
        self.separableconv1 = SeparableConv2d(self.F1*self.D, self.F2, kernel_size=(1, 15), padding=(0, 7), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(self.F2)
        self.averagePooling2 = nn.AvgPool2d(kernel_size=(1,8))
        self.dropout2 = nn.Dropout(p=self.dropoutRate)
        self.flatten1 = nn.Flatten()
        
        self.dense1 = nn.Linear(self.F2*(self.T//32), self.classes)
        
        
    def forward(self, x):
        
        ## block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise1(x)
        x = self.batchnorm2(x)
        x = F.elu(x) #activation
        x = self.averagePooling1(x)
        x = self.dropout1(x)
        
        ## block 2
        x = self.separableconv1(x)
        x = self.batchnorm3(x)
        x = F.elu(x) #activation
        x = self.averagePooling2(x)
        x = self.dropout2(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        
        return x