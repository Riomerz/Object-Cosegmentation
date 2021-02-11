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

kernel = 3
pad = 1
pool_size = 2

encoding_layers = [
    Convolution2D(64,(kernel, kernel), padding ='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(64, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),

    Convolution2D(128, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(128, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),

    Convolution2D(256, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(256, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(256, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),

    Convolution2D(512, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),

    Convolution2D(512, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),
]

decoding_layers = [
    UpSampling2D(size=(pool_size,pool_size)),
    Convolution2D(512, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(size=(pool_size,pool_size)),
    Convolution2D(512,( kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(256, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(size=(pool_size,pool_size)),
    Convolution2D(256, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(256, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(128, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(size=(pool_size,pool_size)),
    Convolution2D(128, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(64, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(size=(pool_size,pool_size)),
    Convolution2D(64, (kernel, kernel), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(n_labels,(1, 1), padding='valid'),
    BatchNormalization(),
]


segnet_basic = models.Sequential()

segnet_basic.add(Layer(input_shape=(img_h, img_w, 3)))


segnet_basic.encoding_layers = encoding_layers
for l in segnet_basic.encoding_layers:
    segnet_basic.add(l)
    print(l.input_shape,l.output_shape,l)

segnet_basic.decoding_layers = decoding_layers
for l in segnet_basic.decoding_layers:
    segnet_basic.add(l)
    print(l.input_shape,l.output_shape,l)

segnet_basic.add(Reshape(( img_h * img_w, n_labels), input_shape=(img_h, img_w, 2)))
#segnet_basic.add(Permute((2, 1)))
segnet_basic.add(Activation('softmax'))

with open('segNet_full_model.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(segnet_basic.to_json()), indent=2))
