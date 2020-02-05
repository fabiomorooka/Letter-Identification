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

    
# This is the main function of this program.    
def main(argv):

	# Initialization of the variables.
    nTrain = 124800  # By default the number of lines will be the maximum one
    nTest = 20800    # By default the number of lines will be the maximum one 
    coef = 0.125      # This is a coefficient to aquire less classes from the database

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
    proportion_trainset = argv[0]
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
    
    print("Finished program")

if __name__ == "__main__":
    main(sys.argv[1:])


