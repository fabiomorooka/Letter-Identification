# **Letter-Identification**

## **Preparing the enviroment for using this project**

### 1. Installing Python (https://www.python.org/)

To check if python is installed on your computer, you must open a terminal and write "python". If you see the python environment, then it is installed, otherwise to install you must write the command "sudo apt get install python2.6" (any version of python2, because then you can upgrade it). So, to update the python, use the command "apt install -only -upgrade python". Finally, to install "pip" which is a python library installation utility, use the command "sudo apt install python-pip".

### 2. Installing Jupyter Lab (https://jupyter.org/)

The process to install jupyter lab is shown in the jupyter website (https://jupyter.org/install). 
With conda, install with the command: "conda install -c conda-forge jupyterlab"
Next, use the command "conda install -c conda-forge notebook" to install the notebook
With pip, install with the command: "pip install jupyterlab"
Next, use the command "pip install notebook" to install the notebook

### 3. Installing python libraries

To install the python libraries that will be used in this project just install it from the "requirements.txt" file.

Execute in the terminal the command (if you are in the main foder of the project, otherwise find the path where the requirements file is located): 

```bash
$ pip install -r ./requirements.txt 
```

## **Using the application**

### 1. To preparete the database that will be used in the study, run the bash file: "init.sh" that is located in the main folder.

On linux just execute the command below and see if the datasets were generated (information of the datasets will be displayed in the terminal)

```bash
$ ./init.sh
```

On windows just click on the bahs file (init.sh)

This project uses jupyter notebook to show the results and the algorithms.

    
## 2. Lunch the jupyter notebook, for doing so just use:
```bash
$ jupyter notebook
``` 
## 3. Using the algorithms of this project:

### 3.1. K-Nearest-Neighbors algorithm created by the students 
The "main.ipynb" file in the "createdAlgorithm" folder analyse the knn classificator created by the students.

### 3.2. Comparing results with the classifiers from scikit-learn

The file "knn.ipynb" analyse just the KNN algorithm, its purpose is to analyse its hyperparameter to chose the best hyperparameter value and use it to classify the handwritten letters.

The file "decisionTree.ipynb" analyse just the Decision Tree algorithm, its purpose is to analyse i#ts hyperparameter to chose the best hyperparameter value and use it to classify the handwritten letters.

The file "randomForest.ipynb" analyse just the Random Forest algorithm, its purpose is to analyse its hyperparameter to chose the best hyperparameter value and use it to classify the handwritten letters.

### 3.3. Using the simple application developped that classify a photo taken by the webcam

The file "knn_app.ipynb" takes the database of photos created and makes the classification of each photo. For doing so it uses the KNN classification model.

The file "knn_app.ipynb" takes the database of photos created and makes the classification of each photo. For doing so it uses the Decision Tree classification model.

The file "knn_app.ipynb" takes the database of photos created and makes the classification of each photo. For doing so it uses the Random Forest classification model.
