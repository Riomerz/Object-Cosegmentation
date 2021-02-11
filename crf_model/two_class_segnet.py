from __future__ import absolute_import
from __future__ import print_function
import os
import scipy
import matplotlib.pyplot as plt

#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu1,floatX=float32,optimizer=None'


import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

from keras import backend as K

import cv2
import numpy as np
import json
np.random.seed(07) # 0bserver07 for reproducibility

img_h = 224
img_w = 224
data_shape = img_h*img_w
n_labels = 2

#class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
class_weighting= [0.0025, 0.001]


def one_hot_it(labels):
    x = np.zeros([224,224,2])
    for i in range(224):
        for j in range(224):
	    #if(labels[i][j] != 1 or labels[i][j] != 0):
	#	x[i,j,1]=1
	   # else:
                x[i,j,labels[i][j]]=1
    return x

# load the data
train_data = np.load('data/train_data_mask.npy')
train_label = np.load('data/train_label_mask.npy')
test_data = np.load('data/test_data_mask.npy')

def plot_results(output):

	for i in range(0,5):
		
		image = train_data[i]
		image = image.reshape(img_h,img_w,3)
		
		#label = test_label[i]
		#label = label.reshape(img_h,img_w,12)
		#label = np.argmax(label, axis= -1)
		tmp = np.squeeze(output[i,:,:,:])
		
		ind = np.argmax(tmp, axis= -1)
	
		r = ind.copy()
		g = ind.copy()
		b = ind.copy()
		#r_gt = label.copy()
		#g_gt = label.copy()
		#b_gt = label.copy()

		Foreground = [255,255,255]
		Background = [0,0,0]

		label_colours = np.array([Background,Foreground])
		for l in range(0,1):
			r[ind==l] = label_colours[l,0]
			g[ind==l] = label_colours[l,1]
			b[ind==l] = label_colours[l,2]
			#r_gt[label==l] = label_colours[l,0]
			#g_gt[label==l] = label_colours[l,1]
			#b_gt[label==l] = label_colours[l,2]

		rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
		rgb_d = np.zeros((ind.shape[0], ind.shape[1], 3))
		rgb = rgb.astype('float32') 
		rgb[:,:,0] = r/255.0
		rgb[:,:,1] = g/255.0
		rgb[:,:,2] = b/255.0
		rgb_d[:,:,0] = r
		rgb_d[:,:,1] = g
		rgb_d[:,:,2] = b
		#rgb_gt = np.zeros((ind.shape[0], ind.shape[1], 3))
		#rgb_gt_d = np.zeros((ind.shape[0], ind.shape[1], 3))
		#rgb_gt[:,:,0] = r_gt/255.0
		#rgb_gt[:,:,1] = g_gt/255.0
		#rgb_gt[:,:,2] = b_gt/255.0
		#rgb_gt_d[:,:,0] = r_gt
		#rgb_gt_d[:,:,1] = g_gt
		#rgb_gt_d[:,:,2] = b_gt

		

		#image = np.transpose(image, (1,2,0))
		#output = np.transpose(output, (1,2,0))
		#image = image[:,:,(2,1,0)]

	
		#scipy.misc.toimage(rgb, cmin=0.0, cmax=255).save('./imgs_results/1/predicted_' +str(i)+'_segnet.png')
		#scipy.misc.toimage(rgb_gt, cmin=0.0, cmax=255).save('./imgs_results/1/label_' +str(i)+'_segnet.png')
		#scipy.misc.toimage(image, cmin=0.0, cmax=255).save('./imgs_results/1/original_' +str(i)+'_segnet.png')
		cv2.imwrite('./imgs_results/2/predicted_' +str(i)+'_segnet.png',rgb_d)
		#cv2.imwrite('./imgs_results/1/label_' +str(i)+'_segnet.png',rgb_gt_d)
		cv2.imwrite('./imgs_results/2/original_' +str(i)+'_segnet.png',image)
		
		image = image/255.0
	
		plt.figure()
		plt.imshow(image,vmin=0, vmax=1)
		plt.figure()
	#plt.imshow(rgb_gt,vmin=0, vmax=1)
	#plt.figure()
		plt.imshow(rgb_d, vmin=0, vmax=1)
		plt.show()
	



train_label_list = []
for i in range(int(train_label.shape[0])):
	#np.savetxt("a.csv", (np.array(train_label[i])), delimiter=",", fmt='%10.5f')
	#print(i)
	#print(train_label[i])
	#print(i)
	train_label_list.append(one_hot_it(train_label[i]))
train_label_list = np.array(train_label_list)
train_label = np.reshape(train_label_list,(422,data_shape,2))

#test_data = np.load('data/test_data.npy')
#test_label = np.load('data/test_label.npy')

#val_data = np.load('data/val_data.npy')
#val_label = np.load('data/val_label.npy')

# load the model:
with open('segNet_basic_model.json') as model_file:
    segnet_basic = models.model_from_json(model_file.read())


segnet_basic.compile(loss="binary_crossentropy", optimizer='sgd', metrics=["accuracy"])

# checkpoint
#filepath="weights.best.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]

nb_epoch = 50
batch_size = 6

# Fit the model
#history = segnet_basic.fit(train_data, train_label, callbacks=callbacks_list, batch_size=batch_size, epochs=nb_epoch,
#                    verbose=1, class_weight=class_weighting, validation_split=0.2, shuffle=True) # validation_split=0.33

# This save the trained model weights to this file with number of epochs
#segnet_basic.save_weights('weights/model_weight_{}.hdf5'.format(nb_epoch))

segnet_basic.load_weights('weights/model_weight_{}.hdf5'.format(nb_epoch))
#segnet_basic.load_weights('weights.best.hdf5')

# Model visualization
#from keras.utils import plot_model
#plot_model(segnet_basic, to_file='model.png', show_shapes=True)

#score = segnet_basic.evaluate(test_data, test_label, verbose=0)
#print ('Test score:', score[0])
#print ('Test accuracy:', score[1])

output = segnet_basic.predict(train_data, verbose=0)
output = output.reshape((output.shape[0], img_h, img_w, n_labels))
#print(output)
plot_results(output)

