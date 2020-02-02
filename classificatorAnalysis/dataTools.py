# -*- coding: utf-8 -*-
# dataTools.py --- Reading binary EMNIST files algorithm

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

import os
import time
import numpy as np
import matplotlib.pyplot as plt

class Letter:
    def __init__(self, data, target):
        self.target = target
        self.width  = int(np.sqrt(len(data)))
        self.image  = data.reshape(self.width, self.width)
        self.features = {'var' : 0,
                         'std' : 0,
                         'mean_grad_M' : 0,
                         'std_grad_M'  : 0,
                         'mean_grad_D' : 0,
                         'std_grad_D'  : 0,
                         'mean_PC_X'   : 0,
                         'std_PC_X'    : 0,
                         'active_PC_X' : 0,
                         'mean_PC_Y'   : 0,
                         'std_PC_Y'    : 0,
                         'active_PC_Y' : 0}
        self.computeFeatures()
    
    def computeFeatures(self):
        # Feature computation
        mag, ang = sobel(self.image)
        pcx, pcy = pixel_count(self.image)
        
        self.features['var'] = np.var(self.image)
        self.features['std'] = np.std(self.image)
        self.features['mean_grad_M'] = np.mean(mag)
        self.features['std_grad_M'] =  np.std(mag)
        self.features['mean_grad_D'] = np.mean(ang)
        self.features['std_grad_D'] =  np.std(ang)
        self.features['mean_PC_X'] =   np.mean(pcx)
        self.features['std_PC_X'] =    np.std(pcx)
        self.features['active_PC_X'] = np.count_nonzero(pcx)
        self.features['mean_PC_Y'] =   np.mean(pcy)
        self.features['std_PC_Y'] =    np.std(pcy)
        self.features['active_PC_Y'] = np.count_nonzero(pcy) 
  
    def __print__(self):
        print("Letter target: "+str(self.target))
        print("Letter features:")
        print(self.features)
        print("Letter image:")
        plt.gray()
        plt.matshow(self.image) 
        plt.show()

def sobel(image):
    w = len(image)
    kernel_x = np.array([ [ 1, 0,-1],
                          [ 2, 0,-2],
                          [ 1, 0,-1] ])

    kernel_y = np.array([ [ 1, 2, 1],
                          [ 0, 0, 0],
                          [-1,-2,-1] ])
    
    grad_x = np.zeros([w - 2, w - 2])
    grad_y = np.zeros([w - 2, w - 2])
    
    for i in range(w - 2):
        for j in range(w - 2):
            grad_x[i, j] = sum(sum(image[i : i + 3, j : j + 3] * kernel_x))
            grad_y[i, j] = sum(sum(image[i : i + 3, j : j + 3] * kernel_y))
            if grad_x[i, j] == 0:
                grad_x[i, j] = 0.000001 
    
    mag = np.sqrt(grad_y ** 2 + grad_x ** 2)
    ang = np.arctan(grad_y / (grad_x + np.finfo(float).eps))
  
    # Gradient computation
  
    return [mag,ang]

def pixel_count(image):
    pc_x = np.zeros(len(image))
    pc_y = np.zeros(len(image))
  
    # Pixel count computation
    for i in range(len(image)):
        pc_x[i] = np.count_nonzero(image[i, :])
        pc_y[i] = np.count_nonzero(image[:, i])

    return [pc_x, pc_y]

class Dataset:
    def __init__(self, array, length):  
        self.array = array
        self.length = length
        self.letters = []
        self.letters = self.createLetters()
        self.raw_features = [[float(f) for f in dig.features.values()] for dig in self.letters]
        self.raw_targets  = [[self.letters[i].target] for i in range(self.length)]
  
    def createLetters(self):
        letters = []
        for row in self.array:
            letters.append(Digit(np.array(row[:-1]), row[-1]))
        return letters

# n : percent of the number of lines in the data base use as array set
def load_data_set(array, n):
    dataset = Dataset(array[:int(n * len(array))], int(n * len(array)))
    
    return dataset

def cvt_obj_nparray(dataset):
    X = np.zeros((dataset.length, 12))
    Y = np.zeros((dataset.length,))
    for i, digit in enumerate(dataset.letters):
        Y[i] = digit.target
        for j, feature in enumerate(digit.features):
            X[i, j] = digit.features[feature]
    return X, Y

def create_data(filename, perc):
    #Load the database (.npy) files 
    img_array = np.load(filename) 

    start_time = time.time()
    print("Creating dataset...")
    print("Number of the lines in the dataset: " + str(len(img_array))) 
    data_set = load_data_set(img_array, perc)
    print("Number of the lines in the dataset: " + str(data_set.length))
    print ("\nFinished creating dataset\n")
    end_time = time.time()

    start_time = time.time()
    X_array, Y_array = cvt_obj_nparray(data_set)
    end_time = time.time()

    return X_array, Y_array

def create_train_data(perc):
    print("Generating TRAIN data...")
    X_train, Y_train = create_data('./../train.npy', perc)
    
    return X_train, Y_train

def create_test_data(perc):
    print("Generating TEST data...")
    X_test, Y_test = create_data('./../test.npy', perc)
    
    return X_test, Y_test

def create_validation_data(perc):
    print("Generating VALIDATION data...")
    X_validation, Y_validation = create_data('./../validation.npy', perc)
    
    return X_validation, Y_validation



