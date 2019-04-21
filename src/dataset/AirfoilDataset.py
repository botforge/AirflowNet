import sys
import os
import os.path as op
import torch
import numpy as np
import scipy
import scipy.ndimage
import re
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import collections
import matplotlib.pyplot as plt
import pdb

class AirfoilDataset(Dataset):
    def __init__(self, data_path, sdf=False, tfms=None):
        self.data_path = data_path
        self.samples = sorted(os.listdir(self.data_path))
        self.sdf = sdf
        self.tfms = tfms
    
    def __len__(self):
        return len(self.samples)

    def get_ground_truth(self, img_tensor, airfoil_mask):
        pressure_range = (-1000, 1000)
        range_diff = pressure_range[1] - pressure_range[0] 
        range_increment = range_diff / 256.0
        pressure_mask = (img_tensor[1, :, :] - 0.5) * range_diff + img_tensor[2, :, :] * range_increment
        # Zero out elements within airfoil
        pressure_mask = pressure_mask * airfoil_mask
        return pressure_mask

    def get_sdf(self, img_tensor, airfoil_mask):
        binmask = airfoil_mask.numpy()
        binmask = binmask.astype(int)
        posfoil = scipy.ndimage.morphology.distance_transform_edt(binmask)
        binmaskcomp = 1 -  binmask
        negfoil = scipy.ndimage.morphology.distance_transform_edt(binmaskcomp)
        sdf = np.subtract(posfoil, negfoil)
        sdf_mask = torch.tensor(sdf).float()
        return sdf_mask

    def get_airfoil_mask(self, img_tensor):
        flattened = torch.sum(img_tensor, dim=0) #Add all three channels
        airfoil_mask = torch.where(flattened == 0, flattened, torch.ones_like(flattened)) #put ones wherever the airfoil is
        return airfoil_mask

    def __getitem__(self, i):
        pressure_img = Image.open(self.data_path + '/' + self.samples[i] + '/' + 'p.png')
        print(self.samples[i])
        size = (800, 800)
        pressure_img.thumbnail(size, Image.NEAREST)
        pressure_img_tensor = transforms.ToTensor()(pressure_img)
        airfoil_mask = self.get_airfoil_mask(pressure_img_tensor) #float tensor
        y = self.get_ground_truth(pressure_img_tensor, airfoil_mask) #float tensor
        
        if self.sdf:
            x = self.get_sdf(pressure_img_tensor, airfoil_mask)
        else:
            x = airfoil_mask

        #find x and y centers
        centerx = int((x.size()[1] // 2))
        centery = int((x.size()[0] // 2) )
        half_size = 128

        #resize and apply transofmrations
        x = x[centery - half_size : centery + half_size, 
                        centerx - half_size:centerx + half_size]
        y = y[centery - half_size : centery + half_size, 
                            centerx - half_size : centerx + half_size]

        if self.tfms is not None:
            x = self.tfms(x)
        return (x, y)

        
            
def main():
    path = '/home/dhruvkar/Desktop/Robotics/rp/AirflowNet/data/images'
    ad = AirfoilDataset(path, sdf=True)
    x, y = ad[2000]
    plt.imshow(y, cmap='gray')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()