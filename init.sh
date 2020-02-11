#!/bin/bash
# init.sh

######################################################
##########      INITIALIZATION BASH      #############
######################################################


# This is a initilization bash that generates all dataset that are used in the study.
# Moreover it generates seconds datasets that can be used for further studies.

# This first python script unzip all the four binary files that contains the EMNIST database,
# Then it generates all numpy files that will be used in the study
echo "Start executing algorithms to create the dataset..."
cd ./EMNINST/
python ./generateDatabase.py
echo "Deleted validation database!"

# This second python script generates the datasets to be used in the study but using 
# the photos taken by the camera as input and not anymore the EMNIST database.

echo "Creating rotated photo letters..."
cd ../application/photos/
python ./dataAugmentation.py 8
echo "Letters created!"
echo "All data created!"
