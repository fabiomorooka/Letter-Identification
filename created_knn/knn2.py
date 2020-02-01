# -*- coding: utf-8 -*-
# knn2.py --- K-Nearest Neighbours (KNN) algorithm

# Copyright (c) 2011-2016  Fabio Morooka <fabio.morooka@gmail.com> and Fernando Amaral <fernando.lucasaa@gmail.com>

# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# code:
from math import sqrt
 
# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    
    return sqrt(distance)
 
# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    
    return neighbors
 
# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    
    return prediction

def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    
    return(predictions)
 
# Test distance function
import numpy as np

def transform_dataset(dataset):
    labels = []
    letters = []
    for row in dataset:
        labels.append(row[-1])
        letters.append(np.array(row[:-1]))
           
    return labels, letters      

print("Loading Train dataset...")
train_array = np.load('./../train.npy')
label_train, img_train = transform_dataset(train_array)
print("Finished loading train dataset!")

print("Loading Test dataset...")
test_array = np.load('./../test.npy')
label_test, img_test = transform_dataset(test_array)
print("Finished loading test dataset!")

print("Loading Validation dataset...")
validation_array = np.load('./../validation.npy')
label_valdation, img_validation = transform_dataset(validation_array)
print("Finished loading validation dataset!")

N = 2000
M = 2000/8

train_x = img_train[:N]
train_y = label_train[:N]
validation_x = img_validation[:M]
validation_y = label_validation[:M]

prediction = predict_classification(train_x, validation_x, 3)
print(prediction)
