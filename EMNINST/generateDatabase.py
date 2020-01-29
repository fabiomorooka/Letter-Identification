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

def remove_file(filename):
    if os.path.isfile(filename):
        print("Removing " + filename)
        os.remove(filename)
    else:
        print(filename + "file does not exist, creating one!")

def remove_folder(foldername):
    if os.path.isdir(foldername):
        print("Removing " + foldername)
        shutil.rmtree(foldername)
    else:
        print(foldername + " folder does not exist, creating one!")

def unzipEMNIST(unzipFolder):
    with zipfile.ZipFile('./emnist-letters.zip', 'r') as zip_file:
        zip_file.extractall(unzipFolder)

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

def separeteDatasets(dataset, perc):
    letters = range(1,27)

    print("Percentage of the testset: " + str(perc))
    print("Percentage of the validationset: " + str(1-perc))

    for letter in letters:
        dl = dataset.loc[dataset[784] == letter]

        test_letters = int(math.ceil(len(dl) * perc))
        validation_letters = len(dl) - test_letters

        df_percTest = dl.head(test_letters)
        df_percValidation = dl.tail(validation_letters)
        
        if letter == 1:
            df_test = df_percTest
            df_validation = df_percValidation
        else:
            df_test = df_test.append(df_percTest)
            df_validation = df_validation.append(df_percValidation)

    print("Number of letters in the testset: " + str(len(df_test)))
    print("Number of letters in the validationset: " + str(len(df_validation)))

    ds_test = df_test.sample(frac=1).reset_index(drop = True) 
    ds_validation = df_validation.sample(frac=1).reset_index(drop = True)  
 
    remove_file('./../test.npy')
    print("Creating test database\n")
    np.save('./../test', ds_test.to_numpy())
    
    remove_file('./../validation.npy')
    print("Creating validation database")
    np.save('./../validation', ds_validation.to_numpy())
    
def main():

    nTrain = 120000  #By default the number of lines will be the maximum one
    nTest = 20500    #By default the number of lines will be the maximum one

    EMNIST_zip_file = "./emnist-letters.zip"
    final_folder = "./emnist-letters"
    train_images_file = "./emnist-letters/emnist-letters-train-images-idx3-ubyte"
    train_labels_file = "./emnist-letters/emnist-letters-train-labels-idx1-ubyte"
    test_images_file = "./emnist-letters/emnist-letters-test-images-idx3-ubyte"
    test_labels_file = "./emnist-letters/emnist-letters-test-labels-idx1-ubyte"

    if not(os.path.isfile(EMNIST_zip_file)):
        print("Verify that the EMNIST zip file is in the folder!")
        sys.exit()
    elif not(os.path.isdir(final_folder)):
        print("Extracting all binary files...")
        unzipEMNIST(final_folder)
        print("Finished unzipping file\n")
        
    proportion_trainset = raw_input('How much do you want to use as trainset?\n')
    while(float(proportion_trainset) <= 0):
        print("\nPercentage must be positive!")
        proportion_trainset = raw_input('Try again: how much do you want to use as trainset?\n')  
    
    if float(proportion_trainset) > 1:
        print("Converting in percentage...")
        proportion_trainset = (float(proportion_trainset) / 100)
        print("Proportion is: " + str(proportion_trainset) + " \n")
    elif float(proportion_trainset) >= 0 and float(proportion_trainset) <= 1:
        pass

    print("Creating train dataset with " + str(nTrain) + " letters...")
    train_list = readBinaryFile(train_images_file, train_labels_file, nTrain)
    train_df = randomizeDataset(train_list)

    remove_file('./../train.npy')
    print("Creating train database...")
    np.save('./../train', train_df.to_numpy())
    print( "Finished train dataset\n")

    print("Creating test and validation dataset with " + str(nTest) + " letters...")
    test_list = readBinaryFile(test_images_file, test_labels_file, nTest)
    test_df = randomizeDataset(test_list) 
    separeteDatasets(test_df, float(proportion_trainset))
    print("Finished creating test and validation dataset\n")
    
    print("Deleting folder with all binary files...")
    remove_folder(final_folder)
    print("Folder deleted!\n")
    
    print("Finished program")

if __name__ == "__main__":
    main()


