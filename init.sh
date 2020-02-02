#!/bin/bash
# init.sh

echo "Start executing algorithms to create the dataset..."
cd ./EMNINST/
python ./generateDatabase.py 30
echo "Deleted validation database!"

echo "Creating rotated photo letters..."
cd ../application/photos/
python ./dataAugmentation.py 1
echo "Letters created!"
echo "All data created!"
