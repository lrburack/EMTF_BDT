#!/bin/bash

# Read the arguments passed to the script
CODE_DIRECTORY=$1
RESULTS_DIRECTORY=$2
NAME=$3

echo "Running job to process: $NAME"
cd $CODE_DIRECTORY
python3 condor_wrapper.py -n $NAME