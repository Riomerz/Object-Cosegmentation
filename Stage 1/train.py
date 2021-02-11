# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 15:11:41 2017

@author: Riomerz
"""

# organize imports
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
#import cPickle
import h5py
import os
import json
#import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import csv

# Files
cwd = os.getcwd()
print(cwd)
DATA_SET_PATH = cwd + '\\Myresult.csv'

# Load the data set for training and testing the logistic regression classifier
#dataset = pd.read_csv(DATA_SET_PATH)
df=pd.read_csv('Myresult.csv', sep=',',header=None)
dataset = df.values
trainData = dataset[:,:256]
trainLabels = dataset[:,256]
test=pd.read_csv('testData.csv', sep=',',header=None)
dataset_test = test.values
testData = dataset_test[:,:256]
testLabels = dataset_test[:,256]
## verify the shape of features and labels
#print ("[INFO] features shape: {}").format(features.shape)
#print ("[INFO] labels shape: {}").format(labels.shape)
#
#print ("[INFO] training started...")
## split the training and testing data
#(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),
#                                                                  np.array(labels),
#                                                                  test_size=test_size,
#                                                                  random_state=seed)
#
#print ("[INFO] splitted train and test data...")
#print ("[INFO] train data  : {}").format(trainData.shape)
#print ("[INFO] test data   : {}").format(testData.shape)
#print ("[INFO] train labels: {}").format(trainLabels.shape)
#print ("[INFO] test labels : {}").format(testLabels.shape)
#
## use logistic regression as the model
print("[INFO] creating model...")
model = LogisticRegression(random_state=None)
model.fit(trainData, trainLabels)


# use rank-1 and rank-5 predictions
print("[INFO] evaluating model...")
#f = open(results, "w")
#rank_1 = 0
#rank_5 = 0
#
## loop over test data
#for (label, features) in zip(testLabels, testData):
#	# predict the probability of each class label and
#	# take the top-5 class labels
#	predictions = model.predict_proba(np.atleast_2d(features))[0]
#	predictions = np.argsort(predictions)[::-1][:5]
#
#	# rank-1 prediction increment
#	if label == predictions[0]:
#		rank_1 += 1
#
#	# rank-5 prediction increment
#	if label in predictions:
#		rank_5 += 1
#
## convert accuracies to percentages
#rank_1 = (rank_1 / float(len(testLabels))) * 100
#rank_5 = (rank_5 / float(len(testLabels))) * 100

## write the accuracies to file
#f.write("Rank-1: {:.2f}%\n".format(rank_1))
#f.write("Rank-5: {:.2f}%\n\n".format(rank_5))

# evaluate the model of cross validation
count_cross = 0
preds_cross = model.predict(trainData)
for i in range(len(trainLabels)):
    if preds_cross[i] == trainLabels[i]:
        count_cross += 1
cross_accuracy = count_cross/len(trainLabels)

# evaluate the model of test data
count = 0
preds = model.predict(testData)
for i in range(len(testLabels)):
    if preds[i] == testLabels[i]:
        count += 1

accuracy = count/len(testLabels)
        

