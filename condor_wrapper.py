import argparse
import pickle
import config
import os
from helpers import build_from_wrapper_dict, get_by_name

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", required=True)
args = parser.parse_args()

NAME = str(args.name)

wrapper_dict = get_by_name(NAME, config.WRAPPER_DICT_NAME)

# Builds the dataset in-place
build_from_wrapper_dict(wrapper_dict)

dict_path = os.path.join(config.RESULTS_DIRECTORY, NAME, config.WRAPPER_DICT_NAME)
# Save the dictionary with the built dataset
with open(dict_path, "wb") as file:
    pickle.dump(wrapper_dict, file)