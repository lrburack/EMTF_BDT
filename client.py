from Dataset import *
import pickle
import os
import argparse
import config
from helpers import build_from_wrapper_dict

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--condor", required=False, default=0)
args = parser.parse_args()

CONDOR = bool(args.condor)

# You may need to request access to these folders
# base_dirs = ["/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_negEndcap_14_0_12_BDT2024/240816_213753_wHMT_wrongDistribution/0000/", "/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_posEndcap_14_0_12_BDT2024/240816_213723_wHMT_wrongDistribution/0000/"]
base_dirs = ["/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_posEndcap_14_0_12_BDT2024/240826_193940/0000", "/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_negEndcap_14_0_12_BDT2024/240826_193901/0000"]

name = "Tests/like_previous_code4"

# Make sure you don't accidently overwrite an existing dataset
if os.path.exists(os.path.join(config.RESULTS_DIRECTORY, name)) and os.path.isdir(os.path.join(config.RESULTS_DIRECTORY, name)):
    print("A dateset with the name " + name + " has already been initiated.")
    overwrite = ""
    while overwrite not in ["y", "n"]:
        overwrite = input("Overwrite it (y/n)? ").lower()
    
    if overwrite == "n":
        exit()

mode = 15

training_data_builder = Dataset(variables=[GeneratorVariables.for_mode(mode), 
                                           Theta.for_mode(mode),
                                           St1_Ring2.for_mode(mode),
                                           dPhi.for_mode(mode),
                                           dTh.for_mode(mode),
                                           FR.for_mode(mode),
                                           Bend.for_mode(mode),
                                           RPC.for_mode(mode),
                                           OutStPhi.for_mode(mode),
                                           dPhiSum4.for_mode(mode),
                                           dPhiSum4A.for_mode(mode),
                                           dPhiSum3.for_mode(mode),
                                           dPhiSum3A.for_mode(mode)
                                           ],
                                filters=[HasModeFilter()], 
                                shared_info=SharedInfo(mode=mode))

wrapper_dict = {
    'training_data_builder': training_data_builder,
    'base_dirs': base_dirs,
    'files_per_endcap': 1
}

os.makedirs(os.path.join(config.RESULTS_DIRECTORY, name), exist_ok=True)
dict_path = os.path.join(config.RESULTS_DIRECTORY, name, config.WRAPPER_DICT_NAME)

if CONDOR:
    with open(dict_path, 'wb') as file:
        pickle.dump(wrapper_dict, file)
    
    condor_submit_path = os.path.join(config.CODE_DIRECTORY, "condor_wrapper.sub")
    command = "condor_submit " + condor_submit_path + " code_directory=" + config.CODE_DIRECTORY + " results_directory=" + config.RESULTS_DIRECTORY + " name=" + name

    print(command)
    os.system(command)
else:
    # Builds the dataset in-place
    build_from_wrapper_dict(wrapper_dict)

    with open(dict_path, 'wb') as file:
        pickle.dump(wrapper_dict, file)