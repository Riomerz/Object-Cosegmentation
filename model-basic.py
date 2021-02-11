from __future__ import absolute_import
from __future__ import print_function
import os


import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization


from keras import backend as K

import cv2
import numpy as np
import json
np.random.seed(07) # 0bserver07 for reproducibility

img_h = 224
img_w = 224
n_labels = 2
data_shape = img_h*img_w



def create_encoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return [
        #ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        #ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        #ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        #ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
    ]

def create_decoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return[
        #ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
	Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        #ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256,( kernel, kernel), padding='same'),
        BatchNormalization(),
	Activation('relu'),	

        UpSampling2D(size=(pool_size,pool_size)),
        #ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
	Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        #ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size, (kernel, kernel), padding='same'),
        BatchNormalization(),
	Activation('relu'),
    ]




segnet_basic = models.Sequential()

segnet_basic.add(Layer(input_shape=(img_h, img_w, 3)))


segnet_basic.encoding_layers = create_encoding_layers()
for l in segnet_basic.encoding_layers:
    segnet_basic.add(l)
    print(l.input_shape,l.output_shape,l)	

# Note: it this looks weird, that is because of adding Each Layer using that for loop
# instead of re-writting mode.add(somelayer+params) everytime.

segnet_basic.decoding_layers = create_decoding_layers()
for l in segnet_basic.decoding_layers:
    segnet_basic.add(l)
    print(l.input_shape,l.output_shape,l)	

segnet_basic.add(Convolution2D(n_labels, (1, 1), padding='same',))

segnet_basic.add(Reshape((data_shape, n_labels), input_shape=(img_h,img_w,n_labels)))
#segnet_basic.add(Permute((2, 1)))
segnet_basic.add(Activation('softmax'))



# Save model to JSON

with open('segNet_basic_model.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(segnet_basic.to_json()), indent=2))
