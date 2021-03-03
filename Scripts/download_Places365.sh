#!/bin/bash

DATASET_DIR=$1

echo "Starting to download Places-365 Dataset and Semantic Segmentations..."
echo "This might take a while depending on your internet connection, please wait..."

# Download Places365 from official website
wget -nc -P $DATASET_DIR http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
unzip -n $DATASET_DIR/places365standard_easyformat.tar -d $DATASET_DIR
rm $DATASET_DIR/places365standard_easyformat.tar

# Download Semantic Segmentations
wget -O $DATASET_DIR/Places_extra.zip http://www-vpu.eps.uam.es/publications/SemanticAwareSceneRecognition/places365_standard_extra_val.zip 
unzip -n $DATASET_DIR/Places_extra.zip -d $DATASET_DIR
rm $DATASET_DIR/Places_extra.zip

echo ========================================================================
echo "All done!"
echo ========================================================================
