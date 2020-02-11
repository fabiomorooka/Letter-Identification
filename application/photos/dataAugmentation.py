# -*- coding: utf-8 -*-
# dataAugmentation.py --- Generating test for the classifier

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

import numpy as np
#np.random.bit_generator = np.random._bit_generator
import matplotlib.pyplot as plt
import os
import imageio
import imgaug as ia
import imgaug.augmenters as iaa

import sys
import cv2 as cv
import pandas as pd
import math
import string
from scipy import ndimage

# This function verify if a file exists and if so it will remove it.
def remove_file(filename):
    if os.path.isfile(filename):
        print("Removing " + filename)
        os.remove(filename)
    else:
        print(filename + "file does not exist, creating one!")

# Function to process the input image and return an image in the format to be compare with the EMNIST database
#
# Parameters:
# - path: where the photo is located 
# - image_name: name of the file
# - show: variable to show the intermediate images created in the processing. 1 -> print images, other value -> not print
#
# Return:
# - image in the EMNIST format (28x28 and white letter and black background)

def image_Processing(path, image_name, show):
    
    # read the image (BGR format)
    image = cv.imread(path + "/" + image_name)
    
    # convert to RGB
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    
    if show == 1:
        print("Printing the source image in gray scale:")
        plt.imshow(gray, cmap="gray")
        plt.show()
        
        # histogram calculation
        print("Printing the gray image's histogram")
        histr = cv.calcHist([gray],[0],None,[256],[0,256])
        plt.plot(histr)
        plt.show()

    # create a binary thresholded image
    _, binary = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)

    if show == 1:
        print("Printing the binary image:")
        plt.imshow(binary, cmap="gray")
        plt.show()
    
    # find the contours from the thresholded image
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # calculate the contours' areas
    contours_areas = []
    for cnt in contours:
        contours_areas.append(cv.contourArea(cnt))
        
    # ascending order list
    contours_areas_sorted = np.sort(contours_areas)
    
    # the image captured have two empty spaces, one in the top and another in the bottom,
    # so the two highest areas values is due to these spaces
    best_cnt_area = contours_areas_sorted[-3]
    best_cnt = contours[contours_areas.index(best_cnt_area)]

    # compute the bounding box
    (x, y, w, h) = cv.boundingRect(best_cnt)

    # extract the region of interest (ROI) using the informations of the bounding box
    diceROI = binary[y-40 : y+h+40, x-40 : x+w+40] 
    
    if show == 1:
        print("Printing the region of intest:")
        plt.imshow(diceROI, cmap="gray")
        plt.show()

    # calculate the center of mass, the height and the width of the ROI
    centre_h, centre_w = ndimage.measurements.center_of_mass(diceROI)
    h, w = diceROI.shape
    
    # if image superior to standard (28x28)
    resized_image = []
    if h or w > 28 :
        resized_image = cv.resize(diceROI,(28,28),interpolation=cv.INTER_AREA)
        if show == 1:
            print("Printing the resized image:")
            plt.imshow(resized_image, cmap="gray")
            plt.show()

    # apply a threshold again because the resize change the binary image
    _, binary2 = cv.threshold(resized_image, 50, 255, cv.THRESH_BINARY)
    
    if show == 1:
        print("Printing the image after the second binarization:")
        plt.imshow(binary2, cmap="gray")
        plt.show()

    # apply a dilation to make lines more visiable
    kernel = np.ones((2,2),np.uint8)
    dilation = cv.dilate(binary2,kernel,iterations = 1) 
    
    return dilation

# This function rotates an iamge
def rotate_images(degree):

    print("Creating more images with a rotation of " + str(degree) + " degrees...")

    img = './letters/'
    alphabet = string.ascii_lowercase

    images = []

    for image_name in os.listdir(img): 
        alphabet_letter = image_name[0]
        path_image = img + image_name
        
        if path_image[-3:] == "JPG" or path_image[-3:] == "jpg":
            if alphabet_letter in alphabet:
                alphabet_number = alphabet.index(alphabet_letter) + 1

            image = image_Processing(img , image_name, 0)

            for rot in range(0, 360, degree):
                rotate = iaa.Affine(rotate=rot) # rotate image
                image_rotated = rotate.augment_images([image])[0]

                letter_target = image_rotated.reshape(-1).tolist()
                letter_target.append(float(alphabet_number))

                row = np.asarray(letter_target)

                images.append(row)

    ds_test = pd.DataFrame(images)

    return ds_test

# This function randomize a dataset
def random_dataset(source_df):
    alphabet_numbers = range(1, 27)
    
    for number in alphabet_numbers:
        dn = source_df.loc[source_df[784] == number]

        if number == 1:
            df_train = dn 
        else:
            df_train = df_train.append(dn)

    ds_train = df_train.sample(frac=1).reset_index(drop = True)
    
    remove_file('test_photos.npy')
    print("Creating test photos database\n")
    np.save('test_photos',ds_train.to_numpy())

# This is the main of the program, it makes some verification then generates all dataset of photos
def main(argv):
    # degree: the step of the rotation
    # the images will be create in the folder with their originals

    if len(argv) == 0:
        print("You must insert a degree as argument!\n")
        sys.exit()

    degree = argv[0]
    while(float(degree) <= 0):
        print("\nDegree must be a positive integer!")
        degree = raw_input('Try again: how many degrees step?\n')

    image_df = rotate_images(int(math.ceil(float(degree))))
    random_dataset(image_df)

    print("Finished creating databases!")

# Main called when called the file
if __name__ == "__main__":
    main(sys.argv[1:])