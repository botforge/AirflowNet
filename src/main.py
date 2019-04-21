import sys
sys.path.append("/home/dhruvkar/Desktop/Robotics/rp/Airflownet/src")
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
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
import time

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

#Change this method to show the same plot
def log_batch_input(x, y, epo, cmap='plasma'):
    for i in range(x.size()[0]):
        fig = plt.figure()
        ax = []
        ax.append(fig.add_subplot(1, 2, 1))
        plt.imshow(x[i, :, :, :], cmap=cmap)
        plt.colorbar()
        ax.append(fig.add_subplot(1, 2, 2))
        plt.imshow(y[i, :,  :, :], cmap=cmap)
        plt.colorbar()
        label = 'x,y' + str(i+1) + ' epo:'+ str(epo)
        writer.add_figure(label, fig)


#Change this method to show the same plot
def log_batch_output(x, y, y_hat, epo, cmap='gray'):
    x = torch.squeeze(x, dim = 1).detach()
    y = torch.squeeze(y, dim = 1).detach()
    y_hat = y_hat.detach()
    for i in range(x.size()[0]):
        fig = plt.figure()
        ax = []
        ax.append(fig.add_subplot(1, 2, 1))
        plt.imshow(y[i, :, :], cmap=cmap)
        plt.colorbar()
        ax.append(fig.add_subplot(1, 2, 2))
        plt.imshow(y_hat[i, :, :], cmap=cmap)
        plt.colorbar()
        label = 'y,y_hat' + str(i+1) + ' epo:'+ str(epo)
        writer.add_figure(label, fig)

def main():
    path = '/home/dhruvkar/Desktop/Robotics/rp/AirflowNet/data/images'

    #Create a subset of the DataSet
    sample_i = range(6)
    airflow_dataset = Subset(AirfoilDataset(path, sdf=False), sample_i)
    airflow_dataloader = DataLoader(airflow_dataset, batch_size=3, num_workers=2, shuffle=True)
    
    #Define Model, Optimizer, Criterion
    au256 = Airflow_Unet256((1, 256, 256))
    crietrion = nn.MSELoss()
    optimizer = optim.SGD(au256.parameters(), lr=1e-2, momentum=0.9)

    saving_index = 0
    for epo in range(1):
        saving_index += 1
        index = 0
        epo_loss = 0
        start = time.time()
        for item in airflow_dataloader:
            index +=1
            start = time.time()
            x = item[0]
            y = item[1]
            x = torch.autograd.Variable(x)
            y = torch.autograd.Variable(y)
            
            optimizer.zero_grad()
            y_hat = au256(x)
            loss = crietrion(y_hat, y)
            loss.backward()

            log_batch_output(x, y, y_hat, epo)
            writer.add_text('Loss', str(loss))

            iter_loss = loss.item()
            epo_loss += iter_loss
            optimizer.step()


    writer.close()
if __name__ == "__main__":
    main()