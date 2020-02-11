# -*- coding: utf-8 -*-
# knn.py --- K-Nearest Neighbours (KNN) algorithm

# Copyright (c) 2019-2020  Fabio Morooka <fabio.morooka@gmail.com> and Fernando Amaral <fernando.lucasaa@gmail.com>

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
import scipy.spatial.distance as distLib
from math import sqrt
import numpy as np

# Function to calculate the Euclidean distance between two vectors
#
# Parameters:
# - row1 and row2: vectors of numbers that represent the object to be calculated
# Return:
# - The euclidean distance
def euclidean_distance(row1, row2):   
    return distLib.euclidean(row1, row2)

# Function to calculate the Manhattan distance between two vectors
#
# Parameters:
# - row1 and row2: vectors of numbers that represent the object to be calculated
#
# Return:
# - The manhattan distance
def manhattan_distance(row1, row2):
    return distLib.cityblock(row1, row2) 

# Function that calculates the N neighbors closer to a specific object that will be classified
#
# Parameters:
# - dataset: The entire dataset that is used to calculate the distance
# - line: The vector that will be classified
# - num_neighbors: The number of nearest neighbors (Default = 5)
# - norm: The norm that will be used (Default = l2)
#
# Return:
# - neighbors: a list of neighbors of a determinate object that will be classified 
def get_neighbors(dataset, line, num_neighbors = 5, norm = 'l2'):
    distances = list()
    for row in dataset:
        if norm == 'l1':
            dist = manhattan_distance(line, row)
        elif norm == 'l2':
            dist = euclidean_distance(line, row)
        distances.append((row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    
    return neighbors

# Function that predict the object.
#
# Parameters:
# - dataset: The entire dataset that is used to calculate the distance
# - line: The vector that will be classified
# - num_neighbors: The number of nearest neighbors (Default = 5)
# - norm: The norm that will be used (Default = l2)
#
# Return:
# - prediction: the prediction label of the object that was classified
def predict_classification(dataset, line, num_neighbors, norm):
    neighbors = get_neighbors(dataset, line, num_neighbors, norm)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    
    return prediction

# Function that predict a set of objects.
#
# Parameters:
# - dataset: The entire dataset that is used to calculate the distance
# - test: All objects that are going to be classified
# - num_neighbors: The number of nearest neighbors (Default = 5)
# - norm: The norm that will be used (Default = l2)
#
# Return:
# - predictions: a list of the prediction label of each object that was classified
def k_nearest_neighbors(dataset, test, num_neighbors, norm):
    predictions = list()
    for row in test:
        output = predict_classification(dataset, row, num_neighbors, norm)
        predictions.append(output)
    
    return(predictions)
