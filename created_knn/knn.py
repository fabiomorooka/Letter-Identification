# -*- coding: utf-8 -*-
# knn.py --- K-Nearest Neighbours (KNN) algorithm

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
import numpy as np

 
# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    
    return sqrt(distance)

# calculate the Manhattan distance between two vectors
def manhattan_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += abs((row1[i] - row2[i]))
    
    return distance 

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors = 5, norm = 'l2'):
    distances = list()
    for train_row in train:
        if norm == 'l1':
            dist = manhattan_distance(test_row, train_row)
        elif norm == 'l2':
            dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    
    return neighbors
 
# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors, norm):
    neighbors = get_neighbors(train, test_row, num_neighbors, norm)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    
    return prediction

def k_nearest_neighbors(train, test, num_neighbors, norm):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors, norm)
        predictions.append(output)
    
    return(predictions)
 

# Test distance function
def predict_letters(N, M, norm):
    print("Loading Train dataset...")
    train_array = np.load('./../train.npy')
    print("Train database has " + str(len(train_array)) + " letters")
    print("Finished loading train dataset!\n")

    print("Loading Test dataset...")
    test_array = np.load('./../test.npy')
    print("Test database has " + str(len(test_array)) + " letters")
    print("Finished loading test dataset!\n")

    print("Loading Validation dataset...")
    validation_array = np.load('./../validation.npy')
    print("Validation database has " + str(len(validation_array)) + " letters")
    print("Finished loading validation dataset!")

    answer = k_nearest_neighbors(train_array[:N], validation_array[:M], 5, norm)

    return answer
