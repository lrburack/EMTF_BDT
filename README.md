# Tools for EMTF pT assignment studies


For the remainder of Run3, a lookup table will be used by the Endcap Muon Track Finder (EMTF) to quickly predict  

The primary purpose of this codebase is to extract training data from ROOT files (EMTFNtuples) for use in developing and testing hypothetical pT assignment engines for the EMTF.


All of the code for building the dataset is in Dataset.py. The wrappers, client, and example are all things I have built around it.

If you want to add or change a feature, make a new class which extends the TrainingVariable. This class must define an __init__ method and a calculate method. 

## Background
The Large Hadron Collider currently collides bunches of protons every 25ns at the Compact Muon Solenoid (CMS) experiment. Only a small fraction of these collisions result in interesting events which can be used for physics analyses. Saving everything would mean saving several pedabytes of data every second, so the CMS detector must make quick decisions about what data to keep, and what to discard.

The CMS trigger system attempts to identify and save the most interesting 1000 events out of the 40 million candidates which occur every second. The Level 1 (L1) trigger is implemented in firmware, capable of dealing with high volume, and reduces the rate from 40MHz to 100KHz. The High Level Trigger (HLT) is implemented in software, and does a much deeper analysis and reconstruction of the data to decide whether to keep the event.

The Endcap Muon Track Finder is part of the L1 trigger. Its job is to identify interesting muons. In most subsystem, interesting means high 
