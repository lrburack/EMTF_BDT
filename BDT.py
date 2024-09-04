from ROOT import TChain, TFile, TH1D, TTree 
from subprocess import Popen, PIPE
import numpy as np
import os
from Compressor import Compressor
from Run3_Variables import *
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import psutil
from multiprocessing import Pool
import argparse
from to_TVMA import convert_model
from math import log, sqrt, atan, pi, log2, exp, sqrt
from array import array

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_jobs", required=False)
parser.add_argument("-i", "--index", required = False)
parser.add_argument("-m", "--mode", required = True)
parser.add_argument("-nb", "--newbend", required = False, default=0)
parser.add_argument("-s", "--showers", required = False, default=0) # Provide a number corresponding to the shower protocol
parser.add_argument("-o", "--outpath", required = False, default="test.root")
args = parser.parse_args()

MODE = int(args.mode)
USE_NEWBEND = int(args.newbend)
SHOWER_PROTOCOL = int(args.showers)
OUTPATH = str(args.outpath)
print(MODE)

MAX_FILE = 2
MAX_EVT = 500
DEBUG = False
PRNT_EVT = 10000

#folders = ["/afs/cern.ch/user/n/nhurley/CMSSW_12_3_0/src/EMTF_MC_NTuple_SingleMu_new_neg.root", "/afs/cern.ch/user/n/nhurley/CMSSW_12_3_0/src/EMTF_MC_NTuple_SingleMu_pos_new.root"]
#base_dirs = ["/eos/user/n/nhurley/SingleMu/SingleMuFlatOneOverPt1To1000GeV_Ntuple_fixed__negEndcap_v2/221215_114244/0000/", "/eos/user/n/nhurley/SingleMu/SingleMuFlatOneOverPt1To1000GeV_Ntuple_fixed__posEndcap_v2/221215_111111/"]
# base_dirs = ["/eos/cms/store/user/eyigitba/emtf/L1Ntuples/Run3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_negEndcap_13_3_1_BDT2024_noGEM_10M/240112_135506/0000/", "/eos/cms/store/user/eyigitba/emtf/L1Ntuples/Run3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_posEndcap_13_3_1_BDT2024_noGEM_10M/240112_152929/0000/"]
# base_dirs=["/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_negEndcap_14_0_12_BDT2024/240725_190959/0000/","/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_posEndcap_14_0_12_BDT2024/240726_145852/0000"]
base_dirs=["/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_posEndcap_14_0_12_BDT2024/240826_193940/0000", "/eos/user/p/pakellin/RUN3/crabOut/CRAB_PrivateMC/SingleMuGun_flatOneOverPt1to1000_negEndcap_14_0_12_BDT2024/240826_193901/0000"]

#station-station transitions for delta phi's and theta's
transitions = ["12", "13", "14", "23", "24", "34"]
#modes we want to analyze
EMTF_MODES = [15, 14, 13, 12, 11, 10, 9, 7, 6, 5, 3]

features_collection = []

#map station to the indices of transitions (12, 13, 14, 23, 24, 34)
#                                          (0,   1,  2,  3,  4,  5)
station_transition_map = {
            1:[0, 1, 2],
            2:[0, 3, 4],
            3:[1, 3, 5],
            4: [2, 4, 5]}

evt_tree  = TChain('EMTFNtuple/tree')
shower_tree  = TChain('MuShowerNtuple/tree')

#recursivelh access different subdirectories of given folder from above
file_list = []
for base_dir in base_dirs:
    nFiles = 0
    break_loop = False
    for dirname, dirs, files in os.walk(base_dir):
        if break_loop: break
        for file in files:
            if break_loop: break
            if not '.root' in file: continue
            file_name = "%s/%s" % (dirname, file)
            nFiles   += 1
            print ('* Loading file #%s: %s' % (nFiles, file))
            evt_tree.Add(file_name)
            shower_tree.Add(file_name)
            if nFiles >= MAX_FILE/2: break_loop = True


#Flag for breaking loop if we hit max file limit
break_loop = False

#Data frame containing a list of these feature dictionaries, columns are features, rows are different tracks
X = pd.DataFrame()
Y = np.array([])
W = np.array([])

Y_eta = np.array([])
Y_phi = np.array([])
Y_pt = np.array([])

#we will want to break the loop when debugging and look at a single entry
event_break = False    
#loop through all events in the input file

nNegEndcap = 0
nPosEndcap = 0

for event in range(evt_tree.GetEntries()):
    if event_break: break
    #if event == MAX_EVT: break
    if event == MAX_EVT and nNegEndcap != 0 and nPosEndcap != 0: 
        break

    if event % PRNT_EVT == 0:
        print('BDT.py: Processing Event #%d' % (event))
        print('Pos-Endcap',nPosEndcap)
        print('Neg-Endcap',nNegEndcap)    
    evt_tree.GetEntry(event)
    shower_tree.GetEntry(event)

    # if evt_tree.emtfTrack_size != 1:
    #     continue

    if(nNegEndcap > MAX_EVT/2 and evt_tree.genPart_eta[0] <= 0):
        continue
    elif(nPosEndcap > MAX_EVT/2 and evt_tree.genPart_eta[0] > 0):
        continue
    #features per track that will be used as inputs to the BDT
    features = Compressor()

    #look at every track in the input file, for-else lol sorry
    track = -1
    for select_track in range(evt_tree.emtfTrack_size):    
        mode = evt_tree.emtfTrack_mode[select_track]
        if mode == MODE or mode == 15: 
            track = select_track
            break
    if track == -1: continue
    features["mode"] = MODE #<-- either truely mode MODE or force to mode MODE

    #only accept the mode we want to train
    if not mode == MODE and not mode == 15: break

    #convert mode to station bit-array representation
    station_isPresent = np.unpackbits(np.array([MODE], dtype='>i8').view(np.uint8))[-4:]
    
    # This code block matches showers with hits along the track
    # This array contains the shower information. The first axis corresponds to the station, the second axis corresponds to the shower type (0: loose, 1: nominal, 2: tight)
    showers_on_track = np.zeros((4, 3), dtype='bool')
    # Loop through each hit. We will check for a corresponding shower
    for station in range(4):
        if not station_isPresent[station]:
            continue
        
        # Each track could have a hit in each station. hitref is used to associate hit information with a partiuclar hit in a track 
        # emtfTrack_hitref<i>[j] tells you where to find information about a hit in station i for track j 
        hitref = getattr(evt_tree, "emtfTrack_hitref" + str(station + 1))[track]

        # Loop through each shower and see if it corresponds to a hit in the track
        for i in range(shower_tree.CSCShowerDigiSize):
            # Check that the hit location matches the shower location
            if (evt_tree.emtfHit_chamber[hitref] == shower_tree.CSCShowerDigi_chamber[i] and 
                evt_tree.emtfHit_ring[hitref] == shower_tree.CSCShowerDigi_ring[i] and 
                evt_tree.emtfHit_station[hitref] == shower_tree.CSCShowerDigi_station[i] and 
                evt_tree.emtfHit_endcap[hitref] == shower_tree.CSCShowerDigi_endcap[i]):
                # Add the shower information to the array
                showers_on_track[station, :] = np.array([shower_tree.CSCShowerDigi_oneLoose[i], shower_tree.CSCShowerDigi_oneNominal[i], shower_tree.CSCShowerDigi_oneTight[i]]).T

    if SHOWER_PROTOCOL == 1:
        for station in range(4):
            if not station_isPresent[station]:
                continue 
            features["looseShower_" + str(station)] = showers_on_track[station][0]
            features["nominalShower_" + str(station)] = showers_on_track[station][1]
            features["tightShower_" + str(station)] = showers_on_track[station][2]
    if SHOWER_PROTOCOL == 2:
        features["looseShowerCount"] = np.sum(showers_on_track[:,0])
    if SHOWER_PROTOCOL == 3:
        features["nominalShowerCount"] = np.sum(showers_on_track[:,1])
    if SHOWER_PROTOCOL == 4:
        features["tightShowerCount"] = np.sum(showers_on_track[:,2])
    if SHOWER_PROTOCOL == 5:
        features["showerBit"] = int(shower_tree.CSCShowerDigiSize >= 1)
    if SHOWER_PROTOCOL == 6:
        features["showerBit"] = int(shower_tree.CSCShowerDigiSize >= 2)

    #define station patterns
    station_pattern = []
    for station in range(4):
        hitref = eval('evt_tree.emtfTrack_hitref%d[%d]' % (station + 1, track))
        features['ph%d' % (station + 1)] = evt_tree.emtfHit_emtf_phi[hitref]
        features['th%d' % (station + 1)] = evt_tree.emtfHit_emtf_theta[hitref]
        if station_isPresent[station]:
            pattern = evt_tree.emtfTrack_ptLUT_cpattern[track][station]
            if not "theta" in features.keys() and station != 0:
                features["theta"] = evt_tree.emtfHit_emtf_theta[hitref]
            for station2 in range(station + 1, 4):
                if station_isPresent[station2]:
                    hitref2 = eval('evt_tree.emtfTrack_hitref%d[%d]' % (station2 + 1, track))
                    features['dTh_' + str(station + 1) + str(station2 + 1)] = evt_tree.emtfHit_emtf_theta[hitref2] - evt_tree.emtfHit_emtf_theta[hitref]
            features['RPC_' + str(station + 1)] = 1 if pattern == 0 else 0 #evt_tree.emtfHit_type[hitref] == 2  maybe?
            #features['bend_' + str(station + 1)] = evt_tree.emtfHit_bend[hitref]

        else:
            pattern = -99
        station_pattern.append(pattern)
        features['pattern_' + str(station + 1)] = pattern
        features['presence_' + str(station + 1)] = station_isPresent[station]

    features['endcap'] = evt_tree.emtfTrack_endcap[track]

    #scalar features
    
    features["st1_ring2"] = evt_tree.emtfTrack_ptLUT_st1_ring2[track]
    #vector features by station
    for station, pattern in enumerate(station_pattern):
        if pattern == -99 or pattern == 10: bend = 0
        if not station_isPresent[station]: continue
        elif pattern % 2 == 0: bend = (10 - pattern) / 2
        elif pattern % 2 == 1: bend = -1 * (11 - pattern) / 2

        if evt_tree.emtfTrack_endcap[track] == 1: bend *= -1

        features["bend_" + str(station + 1)] = bend

        features["FR_" + str(station + 1)] = evt_tree.emtfTrack_ptLUT_fr[track][station]

        #Fix RPC bend
        if features['RPC_' + str(station + 1)] and abs(features["bend_" + str(station + 1)]) == 5: features["bend_" + str(station + 1)] = 0

    #features with station-station transitions
    for i, transition in enumerate(transitions):
        sign = 1 if evt_tree.emtfTrack_ptLUT_signPh[track][i] else -1
        features["dPhi_" + str(transition)] = evt_tree.emtfTrack_ptLUT_deltaPh[track][i] * sign
    
    #clean-up transitions involving not present stations, is this unecessary??
    for i, station_isPresent in enumerate(station_isPresent):
        if not station_isPresent:
            for transition in station_transition_map[i + 1]:
                features["dPhi_" + transitions[transition]] = -999
                features["dTh_" + transitions[transition]] = -999

    for i in range(4):
        if features['presence_' + str(i + 1)]:
            for transition in station_transition_map[i + 1]:
                if features["dPhi_" + str(transitions[transition])] != -999 and "signPhi" not in features.keys():
                    features["signPhi"] = 1 if features["dPhi_" + str(transitions[transition])] >= 0 else -1
                    break

    for i in ['12', '13', '14', '23', '24', '34']:
        features['dPhi_' + str(i)] *= features['signPhi']

    if DEBUG and mode == MODE:
        for k, v in features.items():
            print(k + " = " + str(v))
    
    if (USE_NEWBEND and mode == 15):
        hitref = eval('evt_tree.emtfTrack_hitref1[%d]' % (track))
        if not evt_tree.emtfTrack_ptLUT_st1_ring2[track]: 
            pt=evt_tree.genPart_pt[0]
            me1phi = eval('evt_tree.emtfHit_emtf_phi[%d]' % (hitref))
            sector = evt_tree.emtfHit_sector[hitref]
            me1phi_CMS = (me1phi/60 -7 + (sector-1)*60)*pi/180
            me1theta = eval('evt_tree.emtfHit_emtf_theta[%d]' % (hitref))*pi/180
            me1eta = -log(atan(me1theta/2))
            #print("here")
        else: continue

        dR=[]
        ge1phi_CMSs = []
        #if the hit was in GE11
        for j in range(len(evt_tree.emtfHit_type)):
            
            if evt_tree.emtfHit_type[j] == 3:
                ge1phi = eval('evt_tree.emtfHit_emtf_phi[%d]' % (j))
                ge1phi_CMS = (ge1phi/60 - 7 + (sector-1)*60)*pi/180
                ge1theta = eval('evt_tree.emtfHit_emtf_theta[%d]' % (j))*pi/180
                ge1eta = -log(atan(ge1theta/2))
                dr = sqrt((me1phi_CMS-ge1phi_CMS)**2 + (me1eta-ge1eta)**2)
                ge1phi_CMSs.append(ge1phi_CMS)
                dR.append(dr)
        if (len(dR)>0 and  min(dR)<0.025):
            features['bend_1'] = me1phi_CMS - ge1phi_CMSs[dR.index(min(dR))]

            
    #print("\nCompressing...\n")
    features_precompressed = {k:v for k, v in features.items()}
    #features.compress()
    
    if mode == 15:
        #Get dphi sums, must happen post-compression
        deltaPh_list = [features['dPhi_' + i] for i in transitions]
        features["dPhiSum4"] = sum(deltaPh_list)
        features["dPhiSum4A"] = np.sum(np.abs(deltaPh_list))

        station_deviation = []
        for i in range(4):
            station_deviation += [sum([np.abs(deltaPh_list[transition]) for transition in station_transition_map[i + 1]])]

        outStPh = np.where(station_deviation == max(station_deviation))[0][0] + 1

        if len(np.where(station_deviation == max(station_deviation))[0]) > 1: outStPh = 0

        features["outStPhi"] = outStPh

        if outStPh == 0: outStPh = 1
        
        other_transitions = [i for i in range(6) if i not in station_transition_map[outStPh]]
        features["dPhiSum3"] = sum([deltaPh_list[transition] for transition in other_transitions])
        features["dPhiSum3A"] = sum([abs(deltaPh_list[transition]) for transition in other_transitions])

    if DEBUG and mode == MODE:
        for k, v in features.items():
            print(k + " = " + str(v))

    if(evt_tree.genPart_eta[0] > 0):
        nPosEndcap = nPosEndcap + 1
    if(evt_tree.genPart_eta[0] <= 0):
        nNegEndcap = nNegEndcap + 1

    x_ = {}

    # Add all the features that contain the word shower to x_
    for key in features.keys():
        if "shower" in key.lower():
            x_[key] = features[key]

    for key in Run3TrainingVariables[str(MODE)]:
        x_[key] = features[key]
    if DEBUG:
        with open("/afs/cern.ch/user/n/nhurley/CMSSW_12_3_0/src/EMTFPtAssign2017/inputs.txt", 'r') as compare:
            old_BDT_props = {}
            for i, line in enumerate(compare.readlines()):
                
                prop = line.split(":")[0].strip()

                if "New Track" in line: continue
                value = line.split(":")[-1].strip()
                old_BDT_props[prop] = float(value)

                if "TRK_hit_ids"  in prop: 
                    if old_BDT_props['ph1'] == features['ph1'] and old_BDT_props['ph2'] == features['ph2'] and old_BDT_props['ph3'] == features['ph3'] and old_BDT_props['ph4'] == features['ph4']:
                        matched = True
                        for k, v in x_.items():
                            if v != old_BDT_props[k]:
                                matched = False
                                print("Key: %s, old != new (%d, %d)" % (k, old_BDT_props[k], v))
                        if not matched:
                            print(features_precompressed)
                            print(features)
                            input('check this out!')
                        else: break
                        old_BDT_props = {}
            else:
                print("No match exists!")

    X = pd.concat([X,pd.DataFrame([x_])], ignore_index = True)
    Y = np.append(Y, log(evt_tree.genPart_pt[0]))
    W = np.append(W, 1. / log2(evt_tree.genPart_pt[0] + 0.000001))
    Y_pt = np.append(Y_pt, evt_tree.genPart_pt[0])
    Y_eta = np.append(Y_eta, evt_tree.genPart_eta[0])
    Y_phi = np.append(Y_phi, evt_tree.genPart_phi[0])

seed = 1234

X_train, X_test, Y_train, Y_test, W_train, W_test = train_test_split(X, Y, W, test_size=.5, random_state=123)
X_train_2, X_test_2, Y_train_pt, Y_test_pt, W_train_2, W_test_2 = train_test_split(X, Y_pt, W, test_size=.5, random_state=seed)
X_train_2, X_test_2, Y_train_eta, Y_test_eta, W_train_2, W_test_2 = train_test_split(X, Y_eta, W, test_size=.5, random_state=seed)
X_train_3, X_test_3, Y_train_phi, Y_test_phi, W_train_3, W_test_3 = train_test_split(X, Y_phi, W, test_size=.5, random_state=seed)

dtrain = xgb.DMatrix(data = X_train, label = Y_train, weight = W_train)
dtest = xgb.DMatrix(data = X_test, label = Y_test, weight = W_test)

xg_reg = xgb.XGBRegressor(objective = 'reg:linear', 
                        learning_rate = .1, 
                        max_depth = 5, 
                        n_estimators = 400,
                        max_bins = 1000,
                        nthread = 30)

xg_reg.fit(X_train, Y_train, sample_weight = W_train)


try: outfile = TFile(OUTPATH, 'recreate')
except: outfile = TFile(OUTPATH, 'create')
scale_pt_temp = [0, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 45, 60, 75, 100, 140, 160, 180, 200, 250, 300, 500, 1000] #high-pt range
scale_pt_temp_2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 55, 60] #low-pt range
scale_pt_2  = np.array(scale_pt_temp_2, dtype = 'float64')
scale_pt  = np.array(scale_pt_temp, dtype = 'float64')
h_pt  = TH1D('h_pt_den_EMTF',  '', len(scale_pt_temp) - 1,  scale_pt)
h_pt_trg = TH1D('h_pt_num_EMTF',  '', len(scale_pt_temp)-1,  scale_pt)

h_pt_2  = TH1D('h_pt_den_EMTF_2',  '', len(scale_pt_temp_2) - 1,  scale_pt_2)
h_pt_trg_2 = TH1D('h_pt_num_EMTF_2',  '', len(scale_pt_temp_2)-1,  scale_pt_2)

tree = TTree("TestTree","TestTree")

pt_BDT = array('d', [0])
pt_GEN = array('d', [0])
eta_GEN = array('d', [0])
phi_GEN = array('d', [0])
tree.Branch('pt_BDT', pt_BDT, 'pt_BDT/D')
tree.Branch('pt_GEN', pt_GEN, 'pt_GEN/D')
tree.Branch('eta_GEN', eta_GEN, 'eta_GEN/D')
tree.Branch('phi_GEN', phi_GEN, 'phi_GEN/D')
preds = xg_reg.predict(X_test)
for i, y in enumerate(Y_test):
    pt_real = exp(y)
    pt_pred = exp(preds[i])
    pt_BDT[0] = float(pt_pred)
    pt_GEN[0] = float(pt_real)
    eta_GEN[0] = float(Y_test_eta[i])
    phi_GEN[0] = float(Y_test_phi[i])
    tree.Fill()
    h_pt.Fill(pt_real)
    h_pt_2.Fill(pt_real)
    if pt_pred > 22:
        h_pt_trg.Fill(pt_real)
        h_pt_trg_2.Fill(pt_real)

tree.Write()

h_pt_trg.Divide(h_pt)
h_pt_trg.Write()

h_pt_trg_2.Divide(h_pt_2)
h_pt_trg_2.Write()
del outfile

rmse = np.sqrt(mean_squared_error(Y_test, preds))

print("RMSE: %f" % (rmse))

# input_vars = [(x, 'I') for x in X.head()]
# for idx, tree in enumerate(xg_reg.get_booster().get_dump()):
#    convert_model([tree],itree = idx,input_variables = input_vars, output_xml=f'/afs/cern.ch/user/n/nhurley/BDT/{MODE}/{idx}.xml')