import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
#import argparse
#import common
import numpy as np
import sys
import csv
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from os.path import join
from PIL import Image, ImageOps
#import pandas as pd


 # Parsing input arguments
#parser = argparse.ArgumentParser(description='Feature extraction with imagenet weights')
#parser.add_argument('-i','--image', help='image path to extract features', required=True)
#args = vars(parser.parse_args())

def extract_features(data,train,label):
     # Default directory
    with tf.device('/cpu:0'):
	    cwd = os.getcwd()
	    root_out = cwd
	    
	     # Creating and instanciating the chosen CNN
	    #cnet = common.ConvNet()
	    #cnet.build_resnet50(include_top=False, weights='imagenet', classes=1000)
	    #cnet = VGG16(weights='imagenet', include_top=False)
	    cnet = ResNet50(weights='imagenet', include_top=False)
	     #cnet.build_vgg16(include_top=False, weights='imagenet', classes=1000)
	     #cnet.build_vgg19(include_top=False, weights='imagenet', classes=1000)
	     #cnet.build_inception3(include_top=True, weights='imagenet', classes=1000)
	    
	     # Loading and pre-processing input image
	    #img_path = sys.argv[2]
	    
	    #img = image.load_img('image2.jpg')
	    img = Image.fromarray(data, 'RGB')
	     # ResNet50, VGG16 and VGG19 uses (224,224) while InceptionV3 uses (299,299)
	    img = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)
	    x = image.img_to_array(img)
	    x = np.expand_dims(x, axis=0)
	    
	     # Use preprocess_input() for ResNet50, VGG16 and VGG19, and preprocess_inception() for InceptionV3
	    x = preprocess_input(x)
	     #x = common.preprocess_inception(x)
	    
	    print('Input image shape:', x.shape)
	    
	     # Extracting input image features
	    features_mat = cnet.predict(x, batch_size=1)
	    
	     # Concatenating to 1-D array
	    features = np.ndarray.flatten(features_mat)
	    features = features.reshape(features.shape[0])
	    print("\n[INFO] Output array shape:", features.shape)
	    np.asarray(features)
	    features = np.append(features,label)
	    features.tolist()
	     # Saving output
	    train.append(features)
	    
	    import gc; gc.collect()
