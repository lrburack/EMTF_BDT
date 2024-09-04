#!/bin/bash
echo "running"
cd /afs/cern.ch/user/l/lburack/work/BDTdev/EMTF_BDT

NAME="testrun"

mkdir -p "Results/$NAME"
python3 BDT.py -m 15 -s 0 -o "Results/$NAME/test.root"