# %% import libraries
import helperfunctions_MEG as helper
import plotterfunctions as plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

%matplotlib qt

# %% Plotting settings
plt.ion() # set plots to be non-blocking. (if run in terminal)
# %matplotlib qt # set matplotlib backend (if run in notebook)

# %% Run settings
# raw eeg
loadMEGfresh = True
# visualiseMEG = True
# loadMEGresampled_fresh = False

# full group
run_fullgroup = False

# leave one out cross validation
run_crossvalidation = False

# %% Get metadata
sets = helper.MetaData()
# info = sets.get_eeginfo()
plsc = helper.PLSC()
subs = sets.get_subids()

# %% Load MEG data