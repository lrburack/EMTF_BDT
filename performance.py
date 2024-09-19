import os
import matplotlib.pyplot as plt
import config
import numpy as np
from helpers import get_by_name
import scipy.stats


def getEfficiciencyHist(num_binned, den_binned):
    """
       getEfficiciencyHist creates a binned histogram of the ratio of num_binned and den_binned
       and uses a Clopper-Pearson confidence interval to find uncertainties.

       NOTE: num_binned should be a strict subset of den_binned.

       NOTE: efficiency_binned_err[0] is lower error bar and efficiency_binned_err[1] is upper error bar

       INPUT:
             num_binned - TYPE: numpy array-like
             den_binned - TYPE: numpy array-like
       OUTPUT:
             efficiency_binned - TYPE: numpy array-like
             efficiency_binned_err - TYPE: [numpy array-like, numpy array-like]
       
    """
    # Initializing binned data
    efficiency_binned = np.array([])
    efficiency_binned_err = [np.array([]), np.array([])]

    # Iterating through each bin 
    for i in range(0, len(den_binned)):
        # Catching division by 0 error
        if(den_binned[i] == 0):
            efficiency_binned = np.append(efficiency_binned, 0)
            efficiency_binned_err[0] = np.append(efficiency_binned_err[0], [0])
            efficiency_binned_err[1] = np.append(efficiency_binned_err[1], [0])
            continue

        # Filling efficiency bins
        efficiency_binned = np.append(efficiency_binned, [num_binned[i]/den_binned[i]])

        # Calculating Clopper-Pearson confidence interval
        nsuccess = num_binned[i]
        ntrial = den_binned[i]
        conf = 95.0
    
        if nsuccess == 0:
            alpha = 1 - conf / 100
            plo = 0.
            phi = scipy.stats.beta.ppf(1 - alpha, nsuccess + 1, ntrial - nsuccess)
        elif nsuccess == ntrial:
            alpha = 1 - conf / 100
            plo = scipy.stats.beta.ppf(alpha, nsuccess, ntrial - nsuccess + 1)
            phi = 1.
        else:
            alpha = 0.5 * (1 - conf / 100)
            plo = scipy.stats.beta.ppf(alpha, nsuccess + 1, ntrial - nsuccess)
            phi = scipy.stats.beta.ppf(1 - alpha, nsuccess, ntrial - nsuccess)

        # Filling efficiency error bins
        efficiency_binned_err[0] = np.append(efficiency_binned_err[0], [(efficiency_binned[i] - plo)])
        efficiency_binned_err[1] = np.append(efficiency_binned_err[1], [(phi - efficiency_binned[i])])# - efficiency_binned[i]])

    return efficiency_binned, efficiency_binned_err

names = np.array(["Tests/like_previous_code"])

pt_cut = 22

fig, [low_pt, high_pt] = plt.subplots(2,1)

bins = [0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,
           20,22,24,26,28,30,32,34,36,38,40,42,
           44,46,48,50,60,70,80,90,100,150,200,
           250,300,400,500,600,700,800,900,1000]
x = bins[:-1] + np.diff(bins) / 2

for name in names:    
    prediction_dict = get_by_name(name, config.PREDICTION_NAME)

    predicted_pt = prediction_dict['predicted_pt']
    gen_pt = prediction_dict['gen_data'][:, prediction_dict['gen_features'] == "gen_pt"].squeeze()
    print(np.shape(gen_pt))

    GEN_pt_binned, _ = np.histogram(gen_pt, bins=bins)

    # Count the number of muons in a given GEN_pt bin that were assigned a pT greater than the threshold by the BDT
    a = [np.sum(predicted_pt[np.logical_and(gen_pt > bins[i], gen_pt < bins[i+1])] > pt_cut) for i in range(len(bins) - 1)]

    efficiency_binned, efficiency_binned_err = getEfficiciencyHist(a, GEN_pt_binned)

    efficiency = a / GEN_pt_binned

    stairs_plot = low_pt.scatter(x, efficiency, label=name, s=1)
    color = stairs_plot.get_edgecolor()
    high_pt.scatter(x, efficiency, label=name, color=color, s=1)
    

    low_pt.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)],
                    efficiency_binned, yerr=efficiency_binned_err, xerr=[(bins[i+1] - bins[i])/2 for i in range(0, len(bins)-1)],
                    linestyle="", marker=".", markersize=3, elinewidth = .5, color=color)
    high_pt.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)],
                    efficiency_binned, yerr=efficiency_binned_err, xerr=[(bins[i+1] - bins[i])/2 for i in range(0, len(bins)-1)],
                    linestyle="", marker=".", markersize=3, elinewidth = .5, color=color)


low_pt.set_ylabel("Efficiency")
low_pt.set_xlabel(r"$p_T$")
high_pt.set_ylabel("Efficiency")
high_pt.set_xlabel(r"$p_T$")

low_pt.set_xlim([0,50])

high_pt.set_ylim([.85, 1])

# ax.set_xlim([0,1000])
# ax2.set_xlim([0,1000])
low_pt.legend()

def unique_name(filename, directory="."):
    # Set the initial file path
    base_filepath = os.path.join(directory, filename + ".png")
    filepath = base_filepath
    counter = 1
    # Check if the file exists, and if it does, append a number to make the name unique
    while os.path.exists(filepath):
        filepath = os.path.join(directory, f"{filename}_{counter}.png")
        counter += 1
    
    return filepath

plt.savefig(unique_name("figure", directory = config.FIGURES_DIRECTORY))