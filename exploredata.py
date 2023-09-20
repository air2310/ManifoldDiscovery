import mne 
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# %% Metadata
n_sites = 60

# %% Directory structure
direct = {'dataroot':Path('Data/')}

# %% Generage info structure
# load raw data
fname= direct['dataroot'] / Path("Datasample/trialtest.mat")
epochs=mne.read_epochs_fieldtrip(fname, info=None, data_name='tmp',trialinfo_column=0)

# Create empty info structure
info = mne.create_info(epochs.info['ch_names'], epochs.info['sfreq'], ch_types='eeg')

# Set 10-20 montage
montage2use = mne.channels.make_standard_montage('standard_1020') # mne.channels.get_builtin_montages()
info=info.set_montage(montage2use)
fig = info.plot_sensors(show_names=True, block=False, sphere="eeglab")

# %% Load data

sub = 1
direct['datasub'] = direct['dataroot'] / Path('S' + str(sub) + '/')
plt.ion()

dat = np.empty((n_sites, 59*211))
for site in range(1,n_sites+1):
    # load data
    fname = direct['datasub'] / Path("data_s" + str(sub) + '_site' + str(site) + ".mat")
    epochs=mne.read_epochs_fieldtrip(fname, info=info, data_name='dat',trialinfo_column=0)

    # Baseline correct
    epochs = epochs.apply_baseline(baseline=(-0.1,0))

    # Get average
    evoked = epochs.average()
    evoked_np = evoked.get_data(tmin=0, tmax=1)
    dat[site-1, :] = np.reshape(evoked_np, -1)

    # plot ERP 
    # evoked.plot_joint(times='peaks', title=str(site))

    # get data
    # epochs_np = epochs.get_data()
    # edit events
    # epochs.events[:,2]=site

    # plot epochs
    # epochs.plot_image(picks='CPz')

# %% Feature coding
# Create dictionary to host information
stim_dict = {'site_number':[], 'eccintricity':[], 'visfield_horz':[], 'visfield_vert':[]}

# Set angles for each eccintricity
angles1 = mod([45 - 90*sector for sector in range(4)],360)
eccin = 0
vfh = 1 # 0 = left, 1 = right
vfv = 1 # 0 = bottom, 1 = top
for site in range(1,n_sites+1):
    print(site)
    # site number
    stim_dict['site_number'].append(site)
    
    # eccintricity
    stim_dict['eccin'].append(eccin)
    
    #visfield_horz
    stim_dict['visfield_horz'].append(vfh)

    #visfield_vert
    stim_dict['visfield_horz'].append(vfv)


    # advance
    if np.isin(site, [4, 12, 24, 36, 48, 60]):
        eccin+=1
    if np.isin(site, [2, 4, 8, 12, 18, 24, 30, 36, 42, 48, 54, 60]):
        vfh=np.mod(vfh+1,2)
    if np.isin(site, [1, 3, 6, 10, 15, 21, 27, 33, 39, 45, 51, 57]):
        vfv=np.mod(vfv+1,2)


# %% Trying things
import numpy as np
from sklearn.decomposition import PCA

pca = PCA(n_components=60)
pca.fit(dat)
print(pca.explained_variance_ratio_)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))


plt.figure()
plt.scatter(pca.components_[0],pca.components_[1])
plt.scatter(dat[0],dat[1])

pca = PCA(n_components=2)
X= dat
pca.fit(X)
X_pca = pca.fit_transform(X)
X_pca.shape
plt.figure()
plt.scatter(X_pca[:,0],X_pca[:,1])


plt.figure()
plt.scatter(X[:,0],X[:,1])