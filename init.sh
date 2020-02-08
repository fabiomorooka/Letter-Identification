#!/bin/bash
# init.sh

echo "Start executing algorithms to create the dataset..."
cd ./EMNINST/
python ./generateDatabase.py
echo "Deleted validation database!"

echo "Creating rotated photo letters..."
cd ../application/photos/
python ./dataAugmentation.py 8
echo "Letters created!"
echo "All data created!"
