#!/bin/bash
# remove.sh

######################################################
##########        REMOVE BASH      ###################
######################################################


# This bash is executable only for deleting all dataset 
# .npy files, to make possible to update the Github project 

# Firstly we delete the datasets files

trainFile=./train.npy
testFile=./test.npy
validationFile=./validation.npy

echo "Start Deleting numpy files..."
if [ -f "$trainFile" ];then
	rm -r $trainFile
	echo "Deleted train database!"
else
	echo "Train database does not exist!"
fi

if [ -f "$testFile" ];then
	rm -r $testFile
	echo "Deleted test database!"
else
	echo "Test database does not exist!"
fi

if [ -f "$validationFile" ];then
	rm -r $validationFile
	echo "Deleted validation database!"
else
	echo "Validation database does not exist!"
fi

#Next, we delete the classes files

train_classes_file=./train_classes.npy
test_classes_file=./test_classes.npy
validation_classes_file=./validation_classes.npy

if [ -f "$train_classes_file" ];then
	rm -r $train_classes_file
	echo "Deleted train classes database!"
else
	echo "Train database does not exist!"
fi

if [ -f "$test_classes_file" ];then
	rm -r $test_classes_file
	echo "Deleted test classes database!"
else
	echo "Test database does not exist!"
fi

if [ -f "$validation_classes_file" ];then
	rm -r $validation_classes_file
	echo "Deleted validation classes database!"
else
	echo "Validation database does not exist!"
fi

# Finally we deleate the photos files

photos_file=./application/photos/test_photos.npy

if [ -f "$photos_file" ];then
	rm -r $photos_file
	echo "Deleted photos database!"
else
	echo "Photos database does not exist!"
fi

echo "All files removed!"