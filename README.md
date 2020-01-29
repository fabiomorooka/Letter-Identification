# Letter-Identification

1. Installing Python (https://www.python.org/)

To check if python is installed on your computer, you must open a terminal and write "python". If you see the python environment, then it is installed, otherwise to install you must write the command "sudo apt get install python2.6" (any version of python2, because then you can upgrade it). So, to update the python, use the command "apt install -only -upgrade python". Finally, to install "pip" which is a python library installation utility, use the command "sudo apt install python-pip".

2. Installing Jupyter Lab (https://jupyter.org/)

The process to install jupyter lab is shown in the jupyter website (https://jupyter.org/install). 
With conda, install with the command: "conda install -c conda-forge jupyterlab"
Next, use the command "conda install -c conda-forge notebook" to install the notebook
With pip, install with the command: "pip install jupyterlab"
Next, use the command "pip install notebook" to install the notebook

3. Run "generateDatabase.py" algorithm in the folder "EMNIST". This algorithm will unzip the EMNIST binary files and will read them and generate the ".npy" files. These files will be used as database to do the image classification.

This project uses jupyter notebook to show the results and the algorithms.

4. Using the algorithms of this project:
	
	4.1. Lunch the jupyter notebook 

	4.2. The "knn.ipynb" file in the "createdAlgorithm" folder analyse the knn classificator created by the students.

	4.3. The file "knn.ipynb" analyse just the KNN algorithm, its purpose is to analyse its hyperparameter to chose the best hyperparameter value and use it to classify the handwritten digits from the photos taken.

	4.4. The file "decisionTree.ipynb" analyse just the Decision Tree algorithm, its purpose is to analyse its hyperparameter to chose the best hyperparameter value and use it to classify the handwritten digits from the photos taken.

	4.5. The file "randomForest.ipynb" analyse just the Random Forest algorithm, its purpose is to analyse its hyperparameter to chose the best hyperparameter value and use it to classify the handwritten digits from the photos taken.

	4.6. The file "svm.ipynb" analyse just the Support Vector Machine algorithm, its purpose is to analyse its hyperparameter to chose the best hyperparameter value and use it to classify the handwritten digits from the photos taken.

--------------------------ATENTION!--------------------

After changing the program, before using the github project, please delete the database files (".npy" files)

-------------------------------------------------------

