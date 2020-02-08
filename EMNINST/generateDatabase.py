# -*- coding: utf-8 -*-
# generateDatabase.py --- Reading binary EMNIST files algorithm

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
import sys
import pandas as pd
import numpy as np
import math
import zipfile
import shutil
from collections import Counter
import matplotlib.pyplot as plt

# This function verify if a file exists and if so it will remove it.
def remove_file(filename):
    if os.path.isfile(filename):
        print("Removing " + filename)
        os.remove(filename)
    else:
        print(filename + "file does not exist, creating one!")

# This function verify if a folder exists and if so it will remove it.
def remove_folder(foldername):
    if os.path.isdir(foldername):
        print("Removing " + foldername)
        shutil.rmtree(foldername)
    else:
        print(foldername + " folder does not exist, creating one!")

# This function unzip the EMNIST letters database zip folder in a specific folder.
# The unzipped files are binary files of the database
def unzipEMNIST(unzipFolder):
    with zipfile.ZipFile('./emnist-letters.zip', 'r') as zip_file:
        zip_file.extractall(unzipFolder)

# This function reads the binary files and return the letters (28x28 pixels) with its labbel in the first column. (so 785 values per letter)
def readBinaryFile(imgf, labelf, n):
    f = open(imgf, "rb")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    f.close()
    l.close()

    return images

# This function randomize a list of letters and return a pandas DataFrame. 
def randomizeDataset(imagesList):
    image_list = []
    for row in imagesList:
        new_order = []
        for pixels in row[1:]:
            new_order.append(pixels)
        new_order.append(row[0])
        image_list.append(new_order)

    df = pd.DataFrame(image_list)
    ds = df.sample(frac=1).reset_index(drop = True)

    return ds

# This function deparete a dataset in two: the train dataset and the validation dataset.
def separeteDatasets(dataset, perc, filter_coef):
    letters = range(1,27)
    classesTrain_size = []
    classesValidation_size = []

    print("Percentage of the trainset: " + str(perc))
    print("Percentage of the validationset: " + str(1-perc))

    for letter in letters:
        dl = dataset.loc[dataset[784] == letter]

        total_length = int(len(dl) * filter_coef)

        train_letters = int(math.ceil(total_length * perc))
        validation_letters = total_length - train_letters

        classesTrain_size.append(train_letters)
        classesValidation_size.append(validation_letters)


        df_percTrain = dl.head(train_letters)
        df_percValidation = dl.tail(validation_letters)
        
        if letter == 1:
            df_train = df_percTrain
            df_validation = df_percValidation
        else:
            df_train = df_train.append(df_percTrain)
            df_validation = df_validation.append(df_percValidation)

    ds_train = df_train.sample(frac=1).reset_index(drop = True) 
    ds_validation = df_validation.sample(frac=1).reset_index(drop = True)  
    
    print("------------------[TRAIN DATASET]-----------------------")
    print("Number of letters in the trainset: " + str(len(df_train)))
    balance_train = len(set(classesTrain_size))
    if balance_train == 1:
        print("\nClasses are BALANCED!, with " + str(classesTrain_size[0]) + " sample for class\n")
    else:
        print("\nClasses are UNBALANCED!\n")    

    remove_file('./../train.npy')
    print("Creating train database\n")
    np.save('./../train', ds_train.to_numpy())
    print("--------------------------------------------------------")
    
    print("------------------[VALIDATION DATASET]------------------")
    print("Number of letters in the validationset: " + str(len(df_validation)))
    balance_validation = len(set(classesValidation_size))
    if balance_validation == 1:
        print("\nClasses are BALANCED!, with " + str(classesValidation_size[0]) + " sample for class\n")
    else:
        print("\nClasses are UNBALANCED!\n")

    remove_file('./../validation.npy')
    print("Creating validation database")
    np.save('./../validation', ds_validation.to_numpy())
    print("--------------------------------------------------------")

# This function deparete a dataset in two: the train dataset and the validation dataset.
def reduceDataset(dataset, filter_coef):
    letters = range(1,27)
    classesTest_size = []

    for letter in letters:
        dl = dataset.loc[dataset[784] == letter]

        test_letters = int(len(dl) * filter_coef)        
        classesTest_size.append(test_letters)
        df_percTest = dl.head(test_letters)
        
        if letter == 1:
            df_test = df_percTest
        else:
            df_test = df_test.append(df_percTest)

    print("------------------[TEST DATASET]-----------------------")
    print("Number of letters in the testset: " + str(len(df_test)))

    balance = len(set(classesTest_size))
    if balance == 1:
        print("\nClasses are BALANCED!, with " + str(classesTest_size[0]) + " sample for each class\n")
    else:
        print("\nClasses are UNBALANCED!\n")
    
    ds_test = df_test.sample(frac=1).reset_index(drop = True) 
   
    # Save the test dataset in a ".npy" file
    remove_file('./../test.npy')
    print("Creating test database...")
    np.save('./../test', ds_test.to_numpy())
    print( "Finished test dataset\n")
    print("--------------------------------------------------------")

#--------------------------------------------------------------------------------------------

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
            letters.append(Letter(np.array(row[:-1]), row[-1]))
        return letters

def load_data_set(array):
    dataset = Dataset(array, len(array))
    
    return dataset

def cvt_obj_nparray(dataset):
    X = np.zeros((dataset.length, 12))
    Y = np.zeros((dataset.length,))
    for i, letter in enumerate(dataset.letters):
        Y[i] = letter.target
        for j, feature in enumerate(letter.features):
            X[i, j] = letter.features[feature]
    return X, Y

def create_data_file(filename):
    #Load the database (.npy) files 
    img_array = np.load(filename) 

    print("Creating dataset...")
    data_set = load_data_set(img_array)
    print ("\nFinished creating dataset\n")

    X_array, Y_array = cvt_obj_nparray(data_set)

    return X_array, Y_array

def create_data_list(filename):
    #Load the database (.npy) files 
    img_array = np.load(filename) 

    print("Creating dataset...")
    data_set = load_data_set(img_array)
    print ("\nFinished creating dataset\n")

    return data_set

#Function that normalize the features
def normalize(arr):
    max_line = np.max(arr, axis=0)
    min_line = np.min(arr, axis=0)
    
    arr = (arr - min_line) / (max_line - min_line)
    
    return arr

def create_dataset():
    # Initialization of the variables.
    nTrain = 124800  # By default the number of lines will be the maximum one
    nTest = 20800    # By default the number of lines will be the maximum one 
    coef = 0.0625      # This is a coefficient to aquire less classes from the database

    EMNIST_zip_file = "./emnist-letters.zip"
    final_folder = "./emnist-letters"
    train_images_file = "./emnist-letters/emnist-letters-train-images-idx3-ubyte"
    train_labels_file = "./emnist-letters/emnist-letters-train-labels-idx1-ubyte"
    test_images_file = "./emnist-letters/emnist-letters-test-images-idx3-ubyte"
    test_labels_file = "./emnist-letters/emnist-letters-test-labels-idx1-ubyte"

    # Some cheking if the files are in the coorect folder.
    if not(os.path.isfile(EMNIST_zip_file)):
        print("Verify that the EMNIST zip file is in the folder!")
        sys.exit()
    elif not(os.path.isdir(final_folder)):
        print("Extracting all binary files...")
        unzipEMNIST(final_folder)
        print("Finished unzipping file\n")
        
    # Verifiy if the percentage passed in the first argument is a good argument.
    proportion_trainset = 80
    while(float(proportion_trainset) <= 0):
        print("\nPercentage must be positive!")
        proportion_trainset = raw_input('Try again: how much do you want to use as trainset?\n')  
    
    if float(proportion_trainset) > 1:
        print("Converting in percentage...")
        proportion_trainset = (float(proportion_trainset) / 100)
        print("Proportion is: " + str(proportion_trainset) + " \n")
    elif float(proportion_trainset) >= 0 and float(proportion_trainset) <= 1:
        pass

    # Create the train dataset and validation dataset 
    print("Creating train and validation dataset with " + str(nTrain) + " letters...\n")
    train_list = readBinaryFile(train_images_file, train_labels_file, nTrain)
    train_df = randomizeDataset(train_list)
    separeteDatasets(train_df, float(proportion_trainset), coef)

    # Creating the test  
    print("Creating test with " + str(nTest) + " letters...\n")
    test_list = readBinaryFile(test_images_file, test_labels_file, nTest)
    test_df = randomizeDataset(test_list)
    reduceDataset(test_df, coef) 
       
    # Delete all binary unzipped files to reduce the size of the project
    print("Deleting folder with all binary files...")
    remove_folder(final_folder)
    print("Folder deleted!\n")

def join_data(X_data, Y_data):
    new_array = []
    if len(X_data) == len(Y_data):
        for i in range(len(X_data)):
            line = np.asarray(X_data[i]).reshape(-1).tolist()
            line.append(Y_data[i])
            
            new_array.append(line)

    return pd.DataFrame(new_array)

def create_all_data():
    create_dataset()
    print("Generating TRAIN data...")
    X_train, Y_train = create_data_file('./../train.npy')

    print("Generating TEST data...")
    X_test, Y_test = create_data_file('./../test.npy')

    print("Generating VALIDATION data...")
    X_validation, Y_validation = create_data_file('./../validation.npy')

    X_total = np.concatenate((X_train, X_validation, X_test), axis = 0)
    X_total_norm = normalize(X_total)

    X_train_norm = X_total_norm[0:len(X_train)]
    X_validation_norm = X_total_norm[len(X_train):len(X_train) + len(X_validation)]
    X_test_norm = X_total_norm[-len(X_test):]

    df_test = join_data(X_test_norm, Y_test)
    df_train = join_data(X_train_norm, Y_train)
    df_validation = join_data(X_validation_norm, Y_validation)

    remove_file('./../test_classes.npy')
    print("Created test database")
    np.save('./../test_classes', df_test.to_numpy())

    remove_file('./../train_classes.npy')
    print("Created train database")
    np.save('./../train_classes', df_train.to_numpy())

    remove_file('./../validation_classes.npy')
    print("Created validation database")
    np.save('./../validation_classes', df_validation.to_numpy())

    #return X_train_norm, Y_train, X_test_norm, Y_test, X_validation_norm, Y_validation

def create_train_data_list():
    create_dataset()
    print("Generating TRAIN data...")
    train_list = create_data_list('./../train.npy')
    
    return train_list

def create_test_data_list():
    create_dataset()
    print("Generating TEST data...")
    test_list = create_data_list('./../test.npy')
    
    return test_list

def create_validation_data_list():
    create_dataset()
    print("Generating VALIDATION data...")
    validation_list = create_data_list('./../validation.npy')
    
    return validation_list

def create_train_data():
    create_dataset()
    print("Generating TRAIN data...")
    X_train, Y_train = create_data_file('./../train.npy')
    
    return X_train, Y_train

def create_test_data():
    create_dataset()
    print("Generating TEST data...")
    X_test, Y_test = create_data_file('./../test.npy')
    
    return X_test, Y_test

def create_validation_data():
    create_dataset()
    print("Generating VALIDATION data...")
    X_validation, Y_validation = create_data_file('./../validation.npy')
    
    return X_validation, Y_validation

# This is the main function of this program.    
def main():
    create_all_data()

if __name__ == "__main__":
    main()


