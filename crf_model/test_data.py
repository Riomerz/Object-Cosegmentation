from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
#from tqdm import tqdm
from PIL import Image
import cv2
import scipy.io as sio

import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(cv2.resize(img, (224,224)))
    return images


images = load_images_from_folder('iCoseg/train')
print(len(images))
images = np.array(images)
test_data_mask = np.save('data/test_data_mask',images)
