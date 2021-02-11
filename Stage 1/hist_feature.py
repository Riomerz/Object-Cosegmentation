# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:44:54 2017

@author: Riomerz
"""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import cv2

def hist_feature(img,train,label):
    
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    hist = hist.reshape(hist.shape[0])
    #print (hist)
    #print(label)
    np.asarray(hist)
    hist = np.append(hist,label)
    hist.tolist()

    #print(hist[256])
    
    
    train.append(hist)
    