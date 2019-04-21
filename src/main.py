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


if __name__ == "__main__":
    # Workaround for a deadlock issue on Pytorch 0.2.0: https://github.com/pytorch/pytorch/issues/1838
    multiprocessing.set_start_method('spawn', force=True)
    main()