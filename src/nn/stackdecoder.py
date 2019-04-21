import torch
import torch.nn as nn
import torch.nn.functional as F
from convbnrelu import ConvBnRelu2d
import pdb

class StackDecoder(nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super(StackDecoder, self).__init__()
        padding = (kernel_size - 1)//2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),

            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),

            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1)
        )
    def forward(self,x_big, x):
        #First upsample the output 
        N, C, H, W = x_big.size()
        y = F.upsample(x, size=(H, W), mode='bilinear')
        y = torch.cat([y, x_big], 1) #This step concatenates the initial image to upsampled image (Skip Connection)
        y = self.decode(y)
        return y



    
