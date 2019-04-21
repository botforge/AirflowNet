import torch
import torch.nn as nn
import torch.nn.functional as F
from nn.convbnrelu import ConvBnRelu2d

#Encapsulates a single Stack of down operations, including ConvBnRelu and MaPool
class StackEncoder(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3):
        super(StackEncoder, self).__init__()
        padding = (kernel_size -1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
            
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1)
        )

    def forward(self, x):
        y = self.encode(x)
        y_pool = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_pool

        