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
import collections
import numpy as np


def trainModel(train_x, train_y):
    # do nothing 
    return

def predict(train_x, train_y, test_x, k):
    # create list for distances and targets
    distances = []
    targets = []

    for i in range(len(train_x)):
        # first we compute the euclidean distance
        distance = np.sqrt(np.sum(np.square(test_x - train_x[i])))
        # add it to list of distances
        distances.append([distance, i])

    # sort the list
    distances = sorted(distances)

    # make a list of the k neighbors' targets
    for i in range(k):
        index = distances[i][1]
        #print(y_train[index])
        targets.append(train_y[index])

    # return most common target
    return collections.Counter(targets).most_common(1)[0][0]

# K-Nearest Neighbours algorithm
def kNearestNeighbor(train_x, train_y, test_x, test_y, k):
    # check if k is not larger than n
    if k > len(train_x):
        raise ValueError
    
    # train on the input data
    trainModel(train_x, train_y)

    predictions = []

    # predict for each testing observation
    for i in range(len(test_x)):
        predictions.append(predict(train_x, train_y, test_x[i], k))

    return predictions
