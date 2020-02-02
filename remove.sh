#!/bin/bash
# remove.sh

echo "Start Deleting numpy files..."
rm -r ./train.npy
echo "Deleted train database!"
rm -r ./test.npy
echo "Deleted test database!"
rm -r ./validation.npy
echo "Deleted validation database!"
rm -r ./application/photos/test_photos.npy
echo "Deleted photos database!"
echo "All files removed!"
