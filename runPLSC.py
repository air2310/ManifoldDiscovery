# %% import libraries
import helperfunctions as helper
import plotterfunctions as plot
import matplotlib.pyplot as plt

# %% Plotting settings
plt.ion() # set plots to be non-blocking. (if run in terminal)
%matplotlib qt # set matplotlib backend (if run in notebook)

# %% Run settings
loadEEGfresh = True
visualiseEEG = True

# %% Get metadata
sets = helper.MetaData()

# %% Load EEG data

if loadEEGfresh:
    info = helper.get_eeginfo()


# %% Visualise data
if visualiseEEG
    plot.radialcoords(sets)