# -*- coding: utf-8 -*-
# read.py --- Reading binary EMNIST files algorithm

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
import shutil 
import csv
import sys

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

def changing_columns(filename):

    newFile = "new_" + filename 

    if os.path.isfile(newFile):
        os.remove(newFile)
    else:
        print newFile + " does not exist, so creating one"

    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_NONNUMERIC)
        with open(newFile, "a") as file:
            for row in spamreader:
                    new_row = []
                    for result in row[1:]:
                        new_row.append(result)
                    new_row.append(row[0])

                    writer = csv.writer(file)
                    writer.writerow(new_row)

    shutil.copyfile(newFile, homeDirectory+'/'+filename)
    os.remove(newFile)
    os.remove(filename)

def main():

    nTrainMax = 120000
    nTestMax = 20500
    nTrain = 100000  #By default the number of lines will be the maximum one
    nTest = 20000    #By default the number of lines will be the maximum one
    
    if len(sys.argv) > 3:
    	print "Insert only 2 arguments (the number of TrainLines and TestLines)"
    	sys.exit()

    if os.path.isfile("emnist-letters/emnist-letters-train-images-idx3-ubyte") and os.path.isfile("emnist-letters/emnist-letters-train-labels-idx1-ubyte") and os.path.isfile("emnist-letters/emnist-letters-test-images-idx3-ubyte") and os.path.isfile("emnist-letters/emnist-letters-test-labels-idx1-ubyte"):
        print "All 4 binary files are in the EMNIST folder"
    else:
        print "Verify that all 4 binary files are in the EMNIST folder!"
        sys.exit()
        
    if len(sys.argv) <= 3 and len(sys.argv) > 1:
        if len(sys.argv) == 3:
            nTest = int(sys.argv[2])
    	nTrain = int(sys.argv[1])

    if (nTrain > nTrainMax):
        print "The number of training lines is higher than the maximum one, setting to the maximum value"
    	nTrain = nTrainMax
    
    if (nTest > nTestMax):
        print "The number of testing lines is higher than the maximum one, setting to the maximum value"
    	nTest = nTestMax

    global trainFile, testFile, homeDirectory

    trainFile = "emnist_train.csv"
    testFile = "emnist_test.csv"
    homeDirectory = os.path.split(os.getcwd())[0]

    if os.path.isfile(homeDirectory + '/' + trainFile):
        print "Deleting " + trainFile
        os.remove(homeDirectory + '/' + trainFile)

    if os.path.isfile(homeDirectory + '/' + testFile):
        print "Deleting " + testFile
        os.remove(homeDirectory + '/' + testFile)

    print "Creating train dataset with " + str(nTrain) + " lines"
    convert("emnist-letters/emnist-letters-train-images-idx3-ubyte", "emnist-letters/emnist-letters-train-labels-idx1-ubyte",
        trainFile, nTrain)
    print "Finished train dataset"

    print "Changing columns train file and moving it"
    changing_columns(trainFile)
    print "Finished changing columns and moving it"

    print "Creating test dataset with " + str(nTest) + " lines"
    convert("emnist-letters/emnist-letters-test-images-idx3-ubyte", "emnist-letters/emnist-letters-test-labels-idx1-ubyte",
        testFile, nTest)
    print "Finished test dataset"

    print "Changing columns test file and moving it"
    changing_columns(testFile)
    print "Finished changing columns and moving it"

    print "Finished program"

if __name__ == "__main__":
    main()


