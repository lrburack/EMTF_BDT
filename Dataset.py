import os
import ROOT
import numpy as np
from typing import List, Optional
from datetime import datetime

#station-station transitions for delta phi's and theta's
TRANSITION_NAMES = ["12", "13", "14", "23", "24", "34"]

# TRANSITION_MAP[i] gives the stations corresponding to the transition index i
TRANSITION_MAP = np.array([
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 2],
    [1, 3],
    [2, 3]
])

# STATION_TRANSITION_MAP[i] gives the transition indices corresponding to transitions containing station i
STATION_TRANSITION_MAP = np.array([
    [0, 1, 2],
    [0, 3, 4],
    [1, 3, 5],
    [2, 4, 5]
])

# OUTSTATION_TRANSITION_MAP[i] gives the transition indices corresponding to transitions that don't contain station i
OUTSTATION_TRANSITION_MAP = np.array([
    [3, 4, 5],
    [1, 2, 5],
    [0, 1, 4],
    [0, 1, 3]
])

PRINT_EVENT = 100000

def get_station_presence(mode):
    return np.unpackbits(np.array([mode], dtype='>i8').view(np.uint8)).astype(bool)[-4:]

def transitions_from_mode(mode):
    # In the EMTFNtuple, dPhi and dTheta are 2D arrays. The first dimension is track. The second dimension is a 'transition index'
    # This transition index can be read as the index in this array: ["12", "13", "14", "23", "24", "34"]
    # For a particular mode, we want to know which of these transitions exists. This clever little array operation will get us this
    station_presence = get_station_presence(mode)
    return np.where(np.outer(station_presence, station_presence)[np.logical_not(np.tri(4))])[0]


# -------------------------------    Superclass Definitions    -----------------------------------
class TrainingVariable:
    def __init__(self, feature_names, tree_sources=[], feature_dependencies=[]):
        if isinstance(feature_names, list):
            self.feature_names = feature_names
        else:
            self.feature_names = [feature_names]

        self.tree_sources = tree_sources
        self.feature_dependencies = feature_dependencies

        # Assigned by Dataset class
        self.feature_inds = None
        self.entry_reference = None
        self.shared_reference = None

    def configure(self):
        pass

    def calculate(self, event):
        pass

    def compress(self, event):
        pass

    # This can be overriden to configure variables for a mode. 
    # It must be overriden in the variable constructor takes input
    @classmethod
    def for_mode(cls, mode):
        return cls()


class EventFilter:
    def __init__(self):
        pass
    
    # Return true if the event should be kept, and false otherwise
    def filter(self, event, shared_info):
        return True


class SharedInfo:
    def __init__(self, mode):
        self.mode = mode
        self.station_presence = get_station_presence(mode)
        self.stations = np.where(self.station_presence)[0] # note that these these are shifted to be zero indexed (-1 from station number)
        self.transition_inds = transitions_from_mode(mode)
        self.track = None
        self.hitrefs = np.zeros(len(self.station_presence), dtype='uint8')

        # Set by the Dataset constructor
        self.feature_names = None
        self.entry = None

    def calculate(self, event):
        self.track = None
        modes = np.array(event['EMTFNtuple'].emtfTrack_mode)
        if modes.size == 0:
            return
        good_track_inds = np.where((modes == self.mode) | (modes == 15))[0]

        if good_track_inds.size == 0:
            return
        
        self.track = int(good_track_inds[0])
        
        # hitref is used to associate hit information with a partiuclar hit in a track 
        # emtfTrack_hitref<i>[j] tells you where to find information about a hit in station i for track j
        # Yes, this is ugly, however for performance its quite important to eliminate the use of strings for indexing the root file
        if self.station_presence[0]:
            self.hitrefs[0] = event['EMTFNtuple'].emtfTrack_hitref1[self.track]
        if self.station_presence[1]:
            self.hitrefs[1] = event['EMTFNtuple'].emtfTrack_hitref2[self.track]
        if self.station_presence[2]:
            self.hitrefs[2] = event['EMTFNtuple'].emtfTrack_hitref3[self.track]
        if self.station_presence[3]:
            self.hitrefs[3] = event['EMTFNtuple'].emtfTrack_hitref4[self.track]

    @classmethod
    def for_mode(cls, mode):
        return cls(mode)


# -------------------------------    Dataset Definition    -----------------------------------
class Dataset:
    def __init__(self, variables : List[TrainingVariable], filters : Optional[List[EventFilter]] = [], shared_info : SharedInfo = None, compress : bool=False):
        self.filters = filters
        self.variables = variables
        self.compress = compress

        self.feature_names = []
        for variable in self.variables:
            self.feature_names.extend(variable.feature_names)
        self.num_features = len(self.feature_names)
        self.trainable_features = np.array([not feature_name.startswith("gen") for feature_name in self.feature_names], dtype='bool')

        # Check that each variable has all its required feature dependencies
        for variable in self.variables:
            for feature in variable.feature_dependencies:
                if feature not in self.feature_names:
                    raise Exception(str(type(variable)) + " is missing feature dependency " + feature)

        # This entry variable will be updated for each event. By initializing it now, rather than creating a new array for each event, we save a lot of time
        self.entry = np.zeros(self.num_features, dtype='float32')

        if shared_info == None:
            self.shared_info = SharedInfo()
        else:
            self.shared_info = shared_info

        self.shared_info.feature_names = self.feature_names
        self.shared_info.entry_reference = self.entry

        start_ind = 0
        for variable in self.variables:
            # Give all variables access to the entry so they can use the work of previous variables
            variable.shared_reference = self.shared_info
            # Directly pass the slice within the entry array in which each variable class will deposit features
            variable.feature_inds = self.entry[start_ind : start_ind + len(variable.feature_names)]

            # The way variables are calculated generally depends on the mode and other shared info
            # Once the shared info is shared, we will give the variable a chance 
            # to cache some precalculated information based on the mode
            variable.configure()

            start_ind += len(variable.feature_names)
        
        # Assigned when generate_dataset is called
        self.data = None
        self.filtered = None

    
    def apply_filters(self, event, shared_info):
        # Return true if the event should be kept
        for filter in self.filters:
            if not filter.filter(event, shared_info):
                return False
            
        return True

    def process_event(self, event):
        self.entry.fill(0)
        
        for i, variable in enumerate(self.variables):
            # Each variable will asign its features in entry
            variable.calculate(event)  # Perform the calculation

        if self.compress:
            for variable in self.variables:
                variable.compress(event)

        return self.entry
    
    def build_dataset(self, raw_data):
        tree_names = list(raw_data.keys())

        # Check that all required trees exist in the input data
        for variable in self.variables:
            for source in variable.tree_sources:
                if source not in tree_names:
                    raise Exception(str(type(variable)) + " requires source " + source + " which is not present in the input data.\n Input data has " + str(tree_names))

        # Check that all the trees have the same number of entries
        event_count = raw_data[tree_names[0]].GetEntries()
        for tree in tree_names[1:]:
            if raw_data[tree].GetEntries() != event_count:
                raise Exception("Different number of events in each tree")

        self.data = np.zeros((event_count, self.num_features), dtype='float32')
        self.filtered = np.zeros(event_count, dtype='bool')

        for event_num in range(event_count):
            if event_num % PRINT_EVENT == 0:
                print("* " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\t| Processing event #" + str(event_num))
                print("\t* Trainable event #" + str(np.sum(self.filtered)))
            # Get the current event from each tree.
            # These root objects work in an interesting way. 
            # By calling GetEntry(i) on the entire TChain, the properties associated with that entry become accessible from the TChain object
            for name in tree_names:
                raw_data[name].GetEntry(event_num)

            self.shared_info.calculate(raw_data)

            # apply_filters returns false if the event should be filtered
            if not self.apply_filters(raw_data, self.shared_info):
                continue

            self.filtered[event_num] = True
            self.data[event_num] = self.process_event(raw_data)

        return self.data
    
    def __str__(self):
        print("Dataset object with features: " + self.feature_names)

    def randomize_event_order(self):
        permutation_inds = np.random.permutation(len(self.data))
        self.data = self.data[permutation_inds]
        self.filtered = self.filtered[permutation_inds]

    @staticmethod
    def get_root(base_dirs, files_per_endcap):
        """
        Function to dynamically load TChain objects for all trees in root files,
        concatenating trees within each directory into a single TChain.
        
        Parameters:
        base_dirs (list): List of base directories containing root files.
        files_per_endcap (int): Maximum number of files to load for each endcap.

        Returns:
        dict: Dictionary where keys are directory names, 
            and values are TChain objects with concatenated trees.
        """
        event_data = {}  # Dictionary to store TChains for each directory

        # Iterate through each base directory
        for base_dir in base_dirs:
            nFiles = 0
            break_loop = False
            
            # Recursively traverse through the directory structure
            for dirname, dirs, files in os.walk(base_dir):
                if break_loop: break
                for file in files:
                    if break_loop: break
                    if not file.endswith('.root'): continue  # Only process root files
                    
                    file_name = os.path.join(dirname, file)
                    nFiles += 1
                    print(f'* Loading file #{nFiles}: {file_name}')
                    
                    # Open the ROOT file
                    root_file = ROOT.TFile.Open(file_name)
                    if not root_file or root_file.IsZombie():
                        print(f"Warning: Failed to open {file_name}")
                        continue

                    # Loop through directories in the ROOT file
                    for key in root_file.GetListOfKeys():
                        obj = key.ReadObj()

                        # Check if it's a directory
                        if obj.InheritsFrom("TDirectory"):
                            dir_name = obj.GetName()  # Get the directory name

                            # Add the specific tree from this directory to a TChain
                            tree_name = "tree"  # Assuming the tree name is 'tree' as per original code
                            tree_chain_name = f"{dir_name}/{tree_name}"

                            # Initialize a TChain for this directory if not already present
                            if dir_name not in event_data:
                                event_data[dir_name] = ROOT.TChain(f"{tree_chain_name}")

                            # Add the file to the TChain
                            event_data[dir_name].Add(f"{file_name}/{tree_chain_name}")

                    root_file.Close()

                    # Break the loop if the maximum number of files is reached
                    if nFiles >= files_per_endcap:
                        break_loop = True

        return event_data

# -------------------------------    Filter definitions    -----------------------------------
class HasModeFilter(EventFilter):
    def __init__(self):
        super().__init__()
        pass

    def filter(self, event, shared_info):
        return shared_info.track != None


class TrackCountFilter(EventFilter):
    def __init__(self, track_count = 1):
        super().__init__()
        self.track_count = track_count
        pass

    def filter(self, event, shared_info):
        return event.emtfTrack_size == self.track_count


# -------------------------------    Training variable definitions    -----------------------------------
class GeneratorVariables(TrainingVariable):
    def __init__(self):
        super().__init__(["gen_pt", "gen_eta", "gen_phi"], tree_sources=["EMTFNtuple"])
    
    def calculate(self, event):
        # Directly update the feature slice instead of overwriting it with a new array
        self.feature_inds[0] = event['EMTFNtuple'].genPart_pt[0]
        self.feature_inds[1] = event['EMTFNtuple'].genPart_eta[0]
        self.feature_inds[2] = event['EMTFNtuple'].genPart_phi[0]


class Theta(TrainingVariable):
    def __init__(self, theta_station):
        super().__init__("theta", tree_sources=["EMTFNtuple"])
        self.theta_station = theta_station
        
    def calculate(self, event):
        self.feature_inds[0] = event['EMTFNtuple'].emtfHit_emtf_theta[int(self.shared_reference.hitrefs[self.theta_station])]
    
    @classmethod
    def for_mode(cls, mode):
        # The theta value we use is always the theta in the first station that is not station 1.
        station_presence = get_station_presence(mode)
        return cls(np.argmax(station_presence[1:]) + 1)


class St1_Ring2(TrainingVariable):
    def __init__(self):
        super().__init__("st1_ring2", tree_sources=["EMTFNtuple"])

    def calculate(self, event):
        self.feature_inds[0] = event['EMTFNtuple'].emtfTrack_ptLUT_st1_ring2[self.shared_reference.track]


class dPhi(TrainingVariable):
    def __init__(self, transition_inds):
        self.transition_inds = transition_inds
        # dPhi is defined by two stations, so which dPhi's we train on depends on the mode
        features = ["dPhi_" + TRANSITION_NAMES[ind] for ind in transition_inds]
        super().__init__(features, tree_sources=["EMTFNtuple"])

    def calculate(self, event):
        # for feature_ind, transition_ind in enumerate(self.transition_inds):
        #     sign = 1 if event['EMTFNtuple'].emtfTrack_ptLUT_signPh[int(self.shared_reference.track)][int(transition_ind)] else -1
        #     self.feature_inds[feature_ind] = event['EMTFNtuple'].emtfTrack_ptLUT_deltaPh[int(self.shared_reference.track)][int(transition_ind)] * sign
        # Convert ROOT data to NumPy arrays (if they aren't already NumPy arrays)
        signs = np.array(event['EMTFNtuple'].emtfTrack_ptLUT_signPh[int(self.shared_reference.track)])
        deltaPh = np.array(event['EMTFNtuple'].emtfTrack_ptLUT_deltaPh[int(self.shared_reference.track)])
        
        # Convert signs to -1 or 1 using NumPy's where function
        signs = np.where(signs[self.transition_inds], 1, -1)
        
        # Multiply deltaPh by sign using NumPy array operations
        self.feature_inds[:] = deltaPh[self.transition_inds] * signs * signs[0]
        
    @classmethod
    def for_mode(cls, mode):
        return cls(transitions_from_mode(mode))


class dTh(TrainingVariable):
    # Which transitions to use depending on the mode (based on dTh)
    # Which transitions to use depending on the mode (based on dTh)
    mode_transition_map = {
        15: [2],  # Mode 15 uses dTh_14, so transition 14 (index 2)
        14: [1],  # Mode 14 uses dTh_13, so transition 13 (index 1)
        13: [2],  # Mode 13 uses dTh_14, so transition 14 (index 2)
        11: [2],  # Mode 11 uses dTh_14, so transition 14 (index 2)
        7: [4],   # Mode 7 uses dTh_24, so transition 24 (index 4)
        12: [0],  # Mode 12 uses dTh_12, so transition 12 (index 0)
        10: [1],  # Mode 10 uses dTh_13, so transition 13 (index 1)
        9: [2],   # Mode 9 uses dTh_14, so transition 14 (index 2)
        6: [3],   # Mode 6 uses dTh_23, so transition 23 (index 3)
        5: [4],   # Mode 5 uses dTh_24, so transition 24 (index 4)
        3: [5],   # Mode 3 uses dTh_34, so transition 34 (index 5)
        0: [None] # Mode 0 doesn't use any dTh transitions
    }

    def __init__(self, transition_inds):
        self.transition_inds = transition_inds
        # dTh is defined by two stations, so which dTh's we train on depends on the mode
        features = ["dTh_" + TRANSITION_NAMES[ind] for ind in transition_inds]
        super().__init__(features, tree_sources=["EMTFNtuple"])

    def calculate(self, event):
        for feature_ind, transition_ind in enumerate(self.transition_inds):
            self.feature_inds[feature_ind] = event['EMTFNtuple'].emtfHit_emtf_theta[int(self.shared_reference.hitrefs[TRANSITION_MAP[transition_ind][1]])] - event['EMTFNtuple'].emtfHit_emtf_theta[int(self.shared_reference.hitrefs[TRANSITION_MAP[transition_ind][0]])]
        # Convert the hitrefs and theta values to NumPy arrays
        # hitrefs = self.shared_reference.hitrefs
        # theta_vals = np.array(event['EMTFNtuple'].emtfHit_emtf_theta)
        
        # # Use vectorized NumPy operations to calculate the dTheta for all transitions
        # self.feature_inds[:] = theta_vals[hitrefs[TRANSITION_MAP[self.transition_inds, 1]]] - theta_vals[hitrefs[TRANSITION_MAP[self.transition_inds, 0]]]

    @classmethod
    def for_mode(cls, mode):
        return cls(dTh.mode_transition_map[mode])


class FR(TrainingVariable):
    # Which stations to use depending on the mode
    mode_station_map = {
        15: [0],
        14: [0, 1],
        13: [0, 1],
        11: [0, 2],
        7: [1],
        12: [0, 1],
        10: [0, 2],
        9: [0, 3],
        6: [1, 2],
        5: [1, 3],
        3: [2, 3],
        0: [2, 3],
    }

    def __init__(self, stations):
        self.stations = stations
        features = ["FR_" + str(station + 1) for station in stations]
        super().__init__(features, tree_sources=["EMTFNtuple"])

    def calculate(self, event):
        for feature_ind, station in enumerate(self.stations):
            self.feature_inds[feature_ind] = event['EMTFNtuple'].emtfTrack_ptLUT_fr[int(self.shared_reference.track)][int(station)]

    @classmethod
    def for_mode(cls, mode):
        return cls(FR.mode_station_map[mode])


class RPC(TrainingVariable):
    def __init__(self, stations):
        self.stations = stations
        features = ["RPC_" + str(station + 1) for station in stations]
        super().__init__(features, tree_sources=["EMTFNtuple"])
    
    def calculate(self, event):
        # for feature_ind, station in enumerate(self.stations):
        #     # An RPC was used if the pattern is zero
        #     self.feature_inds[feature_ind] = event['EMTFNtuple'].emtfTrack_ptLUT_cpattern[int(self.shared_reference.track)][int(station)] == 0
        # Convert the cpattern values to a NumPy array
        cpattern_vals = np.array(event['EMTFNtuple'].emtfTrack_ptLUT_cpattern[self.shared_reference.track])
        # Use vectorized NumPy operations to check if the pattern is zero (indicating RPC use)
        self.feature_inds[:] = cpattern_vals[self.stations] == 0

    @classmethod
    def for_mode(cls, mode):
        # There should be an RPC feature for each station in the track
        station_presence = get_station_presence(mode)
        return cls(np.where(station_presence)[0])


class Bend(TrainingVariable):
    # Which stations to use depending on the mode
    mode_station_map = {
        15: [0],
        14: [0],
        13: [0],
        11: [0],
        7: [1],
        12: [0, 1],
        10: [0, 2],
        9: [0, 3],
        6: [1, 2],
        5: [1, 3],
        3: [2, 3],
        0: [],
    }

    pattern_bend_map = np.array([0, -5, 4, -4, 3, -3, 2, -2, 1, -1, 0], dtype='float32')

    def __init__(self, stations):
        self.stations = stations
        features = ["bend_" + str(station + 1) for station in stations]
        super().__init__(features, tree_sources=["EMTFNtuple"])

    def calculate(self, event):
        cpattern_vals = np.array(event['EMTFNtuple'].emtfTrack_ptLUT_cpattern[self.shared_reference.track])[self.stations]
        self.feature_inds[:] = Bend.pattern_bend_map[cpattern_vals] * -1 * event['EMTFNtuple'].emtfTrack_endcap[self.shared_reference.track]

    @classmethod
    def for_mode(cls, mode):
        return Bend(Bend.mode_station_map[mode])

# Classes which extend this will be able to access already calculated dPhi's
class dPhiSum(TrainingVariable):
    def __init__(self, transitions, feature_name="dPhiSum"):
        feature_dependencies = ["dPhi_" + str(transition) for transition in transitions]
        super().__init__(feature_name, feature_dependencies=feature_dependencies)
        self.transitions = transitions
        self.dPhi_reference_inds = None

    def configure(self):
        self.dPhi_reference_inds = np.array([self.shared_reference.feature_names.index("dPhi_" + str(transition)) for transition in self.transitions])

    def calculate(self, event):
        self.feature_inds[0] = np.sum(self.shared_reference.entry_reference[self.dPhi_reference_inds])

    @classmethod
    def for_mode(cls, mode):
        return cls(transitions_from_mode(mode))

# -------------------------------    For use with mode 15    -----------------------------------
class dPhiSum4(dPhiSum):
    def __init__(self):
        super().__init__(TRANSITION_NAMES, feature_name="dPhiSum4")

    @classmethod
    def for_mode(cls, mode):
        if mode != 15:
            raise Exception("dPhiSum4 is for mode 15 only")
        return cls()


class dPhiSum4A(dPhiSum):
    def __init__(self):
        super().__init__(TRANSITION_NAMES, feature_name="dPhiSum4A")
    
    def calculate(self, event):
        self.feature_inds[0] = np.sum(np.abs(self.shared_reference.entry_reference[self.dPhi_reference_inds]))

    @classmethod
    def for_mode(cls, mode):
        if mode != 15:
            raise Exception("dPhiSum4A is for mode 15 only")
        return cls()


class OutStPhi(dPhiSum):
    def __init__(self):
        super().__init__(TRANSITION_NAMES, feature_name="outStPhi")
        self.station_deviations = np.zeros(4, dtype='float32')

    def calculate(self, event):
        dPhis = np.abs(self.shared_reference.entry_reference[self.dPhi_reference_inds])
        self.station_deviations[0] = np.sum(dPhis[STATION_TRANSITION_MAP[0]])
        self.station_deviations[1] = np.sum(dPhis[STATION_TRANSITION_MAP[1]])
        self.station_deviations[2] = np.sum(dPhis[STATION_TRANSITION_MAP[2]])
        self.station_deviations[3] = np.sum(dPhis[STATION_TRANSITION_MAP[3]])
        self.feature_inds[0] = np.argmax(self.station_deviations)

    @classmethod
    def for_mode(cls, mode):
        if mode != 15:
            raise Exception("dPhiSum3 is for mode 15 only")
        return cls()


class dPhiSum3(dPhiSum):
    def __init__(self):
        super().__init__(TRANSITION_NAMES, feature_name="dPhiSum3")
        self.feature_dependencies.extend(["outStPhi"])
        self.out_station_phi_reference = None
    
    def configure(self):
        super().configure()
        self.out_station_phi_reference = self.shared_reference.feature_names.index("outStPhi")

    def calculate(self, event):
        self.feature_inds[0] = np.sum(
            self.shared_reference.entry_reference[self.dPhi_reference_inds[
                OUTSTATION_TRANSITION_MAP[int(self.shared_reference.entry_reference[self.out_station_phi_reference])]
                ]])
    
    @classmethod
    def for_mode(cls, mode):
        if mode != 15:
            raise Exception("dPhiSum3A is for mode 15 only")
        return cls()
    
class dPhiSum3A(dPhiSum):
    def __init__(self):
        super().__init__(TRANSITION_NAMES, feature_name="dPhiSum3A")
        self.feature_dependencies.extend(["outStPhi"])
        self.out_station_phi_reference = None
    
    def configure(self):
        super().configure()
        self.out_station_phi_reference = self.shared_reference.feature_names.index("outStPhi")

    def calculate(self, event):
        self.feature_inds[0] = np.sum(np.abs(
            self.shared_reference.entry_reference[self.dPhi_reference_inds[
                OUTSTATION_TRANSITION_MAP[int(self.shared_reference.entry_reference[self.out_station_phi_reference])]
                ]]))
    
    @classmethod
    def for_mode(cls, mode):
        if mode != 15:
            raise Exception("dPhiSum3 is for mode 15 only")
        return cls()

# -------------------------------    Shower Stuff    -----------------------------------
class ShowerBit(TrainingVariable):
    def __init__(self, shower_threshold):
        super().__init__(feature_names="shower_bit", tree_sources=['MuShowerNtuple'])
        self.shower_threshold = shower_threshold

    def calculate(self, event):
        self.feature_inds[0] = event['MuShowerNtuple'].CSCShowerDigiSize >= self.shower_threshold


# class OutStation(dPhiSum):
#     def __init__(self):
#         super().__init__(TRANSITION_NAMES, feature_name="outStPhi")
    
#     # The out station is the station for which each dPhi transition 
#     def calculate(self, event):
