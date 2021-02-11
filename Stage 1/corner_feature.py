# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:44:54 2017

@author: Riomerz
"""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import argparse
import common
import numpy as np
import sys
import csv
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from os.path import join
from PIL import Image, ImageOps
import pandas as pd

img = cv2.imread('image.jpg',0)
hist = cv2.calcHist([img],[0],None,[256],[0,256])