# %% import libraries
import helperfunctions as helper
import plotterfunctions as plot
import matplotlib.pyplot as plt
import pandas as pd

# %% Plotting settings
plt.ion() # set plots to be non-blocking. (if run in terminal)
# %matplotlib qt # set matplotlib backend (if run in notebook)

# %% Run settings
loadEEGfresh = False
visualiseEEG = False

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

## stack data
# X = [[cond, ch x times]Pid, [cond, ch x times]Pid ..., [cond, ch x times]pidn] .
# Y = [[xpos, ypos, eccin]p1cond1, ...]

# Get positive time values
eeg_resampled = eegdat_df.copy()
eeg_resampled = eeg_resampled.loc[eeg_resampled['time (s)'] > 0, :]
eeg_resampled['time (s)'] = pd.TimedeltaIndex(eeg_resampled['time (s)'],  unit='s')

# resample
grouper = eeg_resampled.groupby(['sub_id', 'site_id', 'ch_name'])
eeg_resampled = grouper.resample('0.01S', on='time (s)', group_keys='True').mean()
eeg_resampled = eeg_resampled.reset_index()


# set behavioural variables
str_behave = ['eccintricity', 'xpos', 'ypos'] #['xpos', 'ypos', 'visfield_horz', 'visfield_vert']

# group
grouper = True
if grouper:
    eeg_resampled = eeg_resampled.groupby(['site_id', 'ch_name', 'time (s)'])['EEG amp. (µV)'].mean().reset_index()
    eeg_resampled['sub_id'] = 'sub1'
    nsubs = 1
else:
    nsubs = 3

# preallocate
X,Y = [], []
for sub in range(nsubs):
    # preallocate for subject
    print('Getting data for subject: ' + str(sub))
    x_n, y_n = [], []

    # cycle through conditions
    for site in range(n_sites):
        # get condition labels
        ydat = stim_df.loc[stim_df.site_number == (site+1), str_behave].to_numpy()
        xdat = eeg_resampled.loc[(eeg_resampled['sub_id'] == ('sub' + str(sub+1))) & (eeg_resampled.site_id == (site+1)), ['EEG amp. (µV)']]

        # resample EEG data
        x_n.append(xdat.to_numpy())
        y_n.append(ydat)

    X.append(np.stack(x_n))
    Y.append(np.stack(y_n))

# save data
eeg_resampled.to_pickle(direct['resultsroot'] / Path('eegretinotopicmappingdf.pkl'))
