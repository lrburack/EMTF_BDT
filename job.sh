#!/bin/bash

# Read the arguments passed to the script
CLUSTER_ID=$1
PROC_ID=$2
NAME=$3
MODE=$4
NEWBEND=$5
SHOWER=$6

echo "Running job with ClusterId: $CLUSTER_ID, ProcId: $PROC_ID, Name: $NAME, Mode: $MODE, Newbend: $NEWBEND"

# Change to the appropriate directory
cd /afs/cern.ch/user/l/lburack/work/BDTdev/EMTF_BDT

# Create the results directory using the NAME
mkdir -p "Results/$NAME"

# Run your Python script with the arguments
python3 BDT.py -m $MODE -nb $NEWBEND -s $SHOWER -o "Results/$NAME"
