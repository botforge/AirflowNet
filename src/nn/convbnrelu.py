import torch
import torch.nn as nn
import torch.nn.functional as F

BN_EPS = 1e-4

#Apply a Convolutional Layer, Batchnorm and Relu in one go
class ConvBnRelu2d(nn.Module):
    
    #Define Net Structure for ease of use
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True, 
    is_relu=True):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)
        if not is_bn: self.bn = None
        if not is_relu: self.relu = None

    def forward(self, x):
        x = self.conv(x) #nn.Conv2d ...
        if self.bn is not None:
            x = self.bn(x)

        if self.relu is not None:
            x = self.relu(x)

        return x