# %% import libraries
import helperfunctions as helper
import plotterfunctions as plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %% Plotting settings
plt.ion() # set plots to be non-blocking. (if run in terminal)
# %matplotlib qt # set matplotlib backend (if run in notebook)

# %% Run settings
loadEEGfresh = False
visualiseEEG = False
loadEEGresampled_fresh = True

# %% Get metadata
sets = helper.MetaData()
info = sets.get_eeginfo()

# %% Load EEG data
# get data
if loadEEGfresh:
    eegdat_df = sets.get_eeg()
else:
    eegdat_df = pd.read_pickle(sets.files['EEGprocessed'])

# %% Visualise EEG data
if visualiseEEG:
    plot.radialcoords(sets)
    plot.ERPs(sets, eegdat_df, info)
    plot.topos(sets, eegdat_df, info)

# %% Prepare data for plsc

# set behavioural variables
str_behave = ['eccintricity', 'xpos', 'ypos']  # ['angle', 'visfield_horz', 'visfield_vert']

# get resampled variables
if loadEEGresampled_fresh:
    eeg_resampled, nsubs = sets.organise_eegdata(eegdat_df, str_behave, grouper=True) # grouper = average data across participants?
else:
    eeg_resampled, nsubs = pd.read_pickle(sets.direct['resultsroot'] / 'eegretinotopicmappingdf.pkl'), 1

# %% stack data up across subjects and conditions to form X and Y
# X = [[cond, ch x times]Pid, [cond, ch x times]Pid ..., [cond, ch x times]pidn] .
# Y = [[cond,[behave values i.e. xpos, ypos]]p1, ..
plsc = helper.PLSC()
X_stack, Y_stack = plsc.stack_data(nsubs, sets, str_behave, eeg_resampled)

# %% # Normalise
# Both X and Y are centered and normalized within each "condition" (subject)
# n (i.e., each Xn and Yn is centered and normalized independently,
# and the sum of squares of a column in one condition is equal to 1,
X_norm, Y_norm = plsc.normalise(X_stack, Y_stack, nsubs, sets)

# %% compute R (covariance)
R = plsc.compute_covariance(X_norm, Y_norm, sets, nsubs)

# %% Restructure variables
U, D, V, X, Y, U_df = plsc.compute_SVD( R, X_norm, Y_norm,sets, str_behave, nsubs)

#%% Compute latent variables
Lx, Ly = plsc.compute_latentvars(sets, X, Y_norm, U_df, U, V, D, nsubs)
# np.savez(sets.direct['resultsroot'] / Path('SVDcomponents.npz'), U=U, D=D,V=V, Lx=Lx, stim_df=setes.stim_df)

# %% Visualise latent vars
sets.nsubs = nsubs
plot.behave_saliances(U_df, sets)
plot.latentspace_2d(sets,Lx)
plot.latentspace_3d(sets, Lx)

# %% Reconstruct EEG

import matplotlib.pyplot as plt
import seaborn as sns

eeg_V = eeg_resampled.loc[(eeg_resampled['sub_id'] == 'sub1') & (eeg_resampled.site_id == 1), :].copy()
for latent in range(V.shape[1]):
    eeg_V['V' + str(latent)] = V[:, latent]

plt.figure()
sns.lineplot(eeg_V, x='time (s)', y='V0', hue='ch_name')

plt.figure()
sns.lineplot(eeg_V, x='time (s)', y='V1', hue='ch_name')

plt.figure()
sns.lineplot(eeg_V, x='time (s)', y='V2', hue='ch_name')