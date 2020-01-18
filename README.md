# Letter-Identification

1. Download the EMNIST letter byte files from EMNIST website (https://www.nist.gov/itl/products-and-services/emnist-dataset). The zip file with all database can be found in: http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip

2. Insert all 4 byte files (emnist-letters-train-images-idx3-ubyte, emnist-letters-train-labels-idx1-ubyte, emnist-letters-test-images-idx3-ubyte and emnist-letters-test-labels-idx1-ubyte) in the EMNINST folder.

3. Run "read.py" algorithm. This algorithm can receive up to 2 arguments, these arguments are respectively the number of letters in the train database and the number of letters in the test database:
- 0 argument: nTrain = 100000 and nTest = 20000
- 1 argument: nTrain = the argument value and nTest = 20000
- 2 arguments: respectively the nTrain and nTest value.
- 3 or more arguments: the algorithm will not execute.

4. Lunch the jupyter notebook "subject.ipynb"

5. After changing the program, before using the github progject, please delete the database and the binary files!
