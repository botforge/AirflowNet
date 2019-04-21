import sys
sys.path.append("/home/dhruvkar/Desktop/Robotics/rp/Airflownet/src")
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt
import pdb
from nn.airflow_unet import Airflow_Unet256
from nn.convbnrelu import ConvBnRelu2d
from nn.stackdecoder import StackDecoder
from nn.stackencoder import StackEncoder
from dataset.AirfoilDataset import AirfoilDataset
from tensorboardX import SummaryWriter
writer = SummaryWriter()

def log_xy(x, y, cmap='plasma'):
    fig = plt.figure()
    ax = []
    ax.append(fig.add_subplot(1, 2, 1))
    plt.imshow(x, cmap=cmap)
    plt.colorbar()
    ax.append(fig.add_subplot(1, 2, 2))
    plt.imshow(y, cmap=cmap)
    plt.colorbar()
    writer.add_figure('x, y', fig)

def main():
    path = '/home/dhruvkar/Desktop/Robotics/rp/AirflowNet/data/images'
    airflow_dataset = AirfoilDataset(path, sdf=False)
    
    airflow_dataloader = DataLoader(airflow_dataset, batch_size=3, shuffle=True, num_workers=2)
if __name__ == "__main__":
    main()