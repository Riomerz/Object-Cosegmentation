# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 18:55:06 2017

@author: Riomerz
"""
import tensorflow as tf
sess = tf.Session()

# import the necessary packages

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
#from extract_features import extract_features
from extract_features import extract_features
#import numpy.core.multiarray

with tf.device('/cpu:0'):
# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required = True, help = "Path to the image")
#args = vars(ap.parse_args())
	train = []

# load the image and apply SLIC and extract (approximately)
# the supplied number of segments
#image = cv2.imread(args["image"])

	for i in range(3,20):
	    image = cv2.imread('images/image' + str(i) + '.jpg')
	    #img = image.load_img('image2.jpg')
	    ground_truth = cv2.imread('ground_truth/ground_truth' + str(i) + '.jpg')
	    segments = slic(img_as_float(image), n_segments = 100, sigma = 5)
	    res = np.zeros(image.shape[:2], dtype = "uint8")
	    selected =  np.zeros(image.shape[:2], dtype = "uint8")
	    res = ground_truth[:,:,0]
	     
	    # show the output of SLIC
	    fig = plt.figure("Superpixels")
	    ax = fig.add_subplot(1, 1, 1)
	    ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments))
	    plt.axis("off")
	    plt.show()
	    label = np.zeros(1)
	    
	    # loop over the unique segment values
	    for (i, segVal) in enumerate(np.unique(segments)):
	    	# construct a mask for the segment
		print ("[x] inspecting segment %d" % (i))
		mask = np.zeros(image.shape[:2], dtype = "uint8")
		#res = np.zeros(image.shape[:2], dtype = "uint8")
		mask[segments == segVal] = 255
	     
	    	# show the masked region

		#cv2.imshow("Mask", mask)
		superpixel_count = np.count_nonzero(mask)
		valid = np.count_nonzero(cv2.bitwise_and(res,mask))
		fraction_covered = (valid/superpixel_count)*100
		if fraction_covered > 60:
		    selected = np.add(selected,mask)
		    label[0] = 1
		    #print(label)
		else:
		    label[0] = 0
		    #print (label)
		#cv2.imshow("Selected", cv2.bitwise_and(res,mask))
		#cv2.imshow("Applied", cv2.bitwise_and(image, image, mask = mask))
		superpixels = cv2.bitwise_and(image, image, mask = mask)

		extract_features(superpixels,train,label)

		#cv2.waitKey(0)
	    #train_ = np.asarray(train)
	np.savetxt("testData.csv", train, delimiter=",")    
	    #cv2.imshow("Segmented",selected)
	    #cv2.waitKey(0)
