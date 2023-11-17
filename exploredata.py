import mne
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Plotting settings
# %matplotlib qt # set matplotlib backend
plt.ion()
%matplotlib qt

# %% Metadata
n_sites = 60
n_electrodes = 59
n_subs = 28

# %% Directory structure
direct = {'dataroot': Path('Data/'), 'resultsroot': Path('Results/')}

# %% Feature coding for each stimulation location
# Create dictionary to host information
stim_dict = {'site_number': [], 'eccintricity': [], 'angle': []}

# loop through sites to assign properties
eccin = 0
for site in range(n_sites):
    # set angle
    if eccin == 0:
        angle = np.mod(45 - 90*site, 360)
    if eccin == 1:
        angle = np.mod(67.5 - 45*(site-4), 360)
    if eccin > 1:
        angle = np.mod(75 - 30*np.mod(site, 12), 360)

    # set properties
    stim_dict['site_number'].append(site+1)  # site number
    stim_dict['angle'].append(angle)  # site number
    stim_dict['eccintricity'].append(eccin)  # eccintricity

    # advance eccintricities
    if np.isin(site+1, [4, 12, 24, 36, 48, 60]):
        eccin += 1

# set visual fields
stim_dict['visfield_horz'] = (np.mod(stim_dict['angle'], 275) < 90).astype(int)
stim_dict['visfield_horz_str'] = np.array(['Left', 'Right'])[stim_dict['visfield_horz']]

stim_dict['visfield_vert'] = (np.array(stim_dict['angle']) < 180).astype(int)
stim_dict['visfield_vert_str'] = np.array(['Bottom', 'Top'])[stim_dict['visfield_horz']]

# get cardinal coordinates
stim_dict['xpos'] = (np.array(stim_dict['eccintricity'] ) + 1) * np.cos(np.deg2rad(stim_dict['angle']))
stim_dict['ypos'] = (np.array (stim_dict['eccintricity']) + 1) * np.sin(np.deg2rad(stim_dict['angle']))

# convert to pandas dataframe
stim_df = pd.DataFrame(stim_dict)

# Plot result
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6,6))
for site in stim_df.index:
    # set colours
    if (stim_df.angle[site] > 0) & (stim_df.angle[site] < 90): col = plt.colormaps['Purples']((stim_df.eccintricity[site]+1)/7)
    if (stim_df.angle[site] > 90) &  (stim_df.angle[site] < 180): col = plt.colormaps['Blues']((stim_df.eccintricity[site]+1)/7)
    if (stim_df.angle[site] > 180) &  (stim_df.angle[site] < 270):col = plt.colormaps['YlOrBr']((stim_df.eccintricity[site]+1)/7)
    if (stim_df.angle[site] > 270) &  (stim_df.angle[site] < 360): col = plt.colormaps['Reds']((stim_df.eccintricity[site]+1)/7)
    ax.plot(np.deg2rad(stim_df.angle[site]), stim_df.eccintricity[site]+1, marker='o', markerfacecolor = col, markeredgecolor = col, markersize=20, alpha=0.75)
    plt.text(np.deg2rad(stim_df.angle[site]), stim_df.eccintricity[site]+1, str(site+1), color=[0.4,0.4,0.4], horizontalalignment='center',
             verticalalignment='center')
ax.set_title("Stimulation positions", va='bottom')
ax.set_rlabel_position(90)  # Move radial labels away from plotted line
ax.set_rlim(0,7)
ax.grid(True)
plt.show(block=False)
plt.savefig(direct['resultsroot'] / Path('RadialCoords.png'))

# %% Generage EEG info structure
# load raw data
fname = direct['dataroot'] / Path("Datasample/trialtest.mat")
epochs = mne.read_epochs_fieldtrip(fname, info=None, data_name='tmp',trialinfo_column=0)

# Create empty info structure
info = mne.create_info(epochs.info['ch_names'], epochs.info['sfreq'], ch_types='eeg')

# Set 10-20 montage
montage2use = mne.channels.make_standard_montage('standard_1020') # mne.channels.get_builtin_montages()
info= info.set_montage(montage2use)
fig = info.plot_sensors(show_names=True, block=False, sphere="eeglab")

# %% Load and store EEG data
# Preallocate
# dat_pca = np.empty((n_sites, n_electrodes*211))
eegdat_dict = {'sub_id':[], 'site_id':[], 'ch_name':[], 'time (s)':[], 'EEG amp. (µV)':[] }

# loop through subjects
for sub in range(n_subs):
    direct['datasub'] = direct['dataroot'] / Path('S' + str(sub+1) + '/')

    # loop through sites
    for site in range(n_sites):
        # load data
        fname = direct['datasub'] / Path("data_s" + str(sub + 1) + '_site' + str(site + 1) + ".mat")
        epochs = mne.read_epochs_fieldtrip(fname, info=info, data_name='dat',trialinfo_column=0)

        # Baseline correct
        epochs = epochs.apply_baseline(baseline=(-0.1,0))

        # Get average
        evoked = epochs.average()

        # store data in a big dictionary
        n_times = len(evoked.times.astype(list))
        tmp = evoked.get_data()
        for ii_elec, electrode in enumerate(info.ch_names):
            eegdat_dict['time (s)'].extend(evoked.times.astype(list))
            eegdat_dict['ch_name'].extend([electrode]*n_times)
            eegdat_dict['sub_id'].extend(['sub' + str(sub+1)]*n_times)
            eegdat_dict['site_id'].extend([(site+1)]*n_times)
            eegdat_dict['EEG amp. (µV)'].extend(tmp[ii_elec,:].astype(list))

# Convert stored trials to dataframe
eegdat_df = pd.DataFrame(eegdat_dict)

# %% Organise some plotting groups we might want to plot by

horz_groups = {'-3':[57,58,45,46], '-2':[34,22,33,21], '-1':[11,4,10,3],
               '1':[1,6,2,7], '2':[15,27,16,28], '3':[39,51,40,52]}

vert_groups = {'-3':[55,54,43,42], '-2':[31,30,19,18], '-1':[9,8,3,2],
               '1':[1,4,5,12], '2':[24,13,36,25], '3':[48,37,60,49]}

# assign horizontal groups
eegdat_df['horz_group'] = 0
stim_df['horz_group'] = 0
for group in horz_groups:
    for index in horz_groups[group]:
        eegdat_df.loc[eegdat_df.site_id==index, 'horz_group'] = int(group)
        stim_df.loc[stim_df.site_number==index, 'horz_group'] = int(group)

# assign vertical groups
eegdat_df['vert_group'] = 0
stim_df['vert_group'] = 0
for group in vert_groups:
    for index in vert_groups[group]:
        eegdat_df.loc[eegdat_df.site_id==index, 'vert_group'] = int(group)
        stim_df.loc[stim_df.site_number==index, 'vert_group'] = int(group)


# %% Plot some results
# %% Step 1 - Different Horizontal and vertical groups ERPs
colormap = "coolwarm" #'coolwarm'

for stim_group in [ 'horz_group','vert_group']: #'vert_group'
    if stim_group == 'horz_group':
        chansplot = ['CPz', 'PO7', 'Oz', 'PO8']
        colormap = sns.color_palette(plt.colormaps["PuOr"](np.linspace(0.2,0.8,6))) #"coolwarm"

    if stim_group == 'vert_group':
        chansplot = ['AFz', 'Cz', 'Pz', 'Oz']
        colormap = 'coolwarm'

    # generate figures
    fig, ax = plt.subplots(2,3, layout='tight', figsize=(16,10))
    # sensors
    info.plot_sensors(show_names=chansplot, block=False, sphere="eeglab", axes = ax[0][0])
    ax[0][0].set_title('Sensor Positions')

    # stimulation locations
    dat_plot = stim_df.loc[stim_df[stim_group]==0]
    sns.scatterplot(data=dat_plot, x='xpos', y = 'ypos', color='grey', s=100, ax=ax[0][2])
    dat_plot = stim_df.loc[stim_df[stim_group]!=0]
    sns.scatterplot(data=dat_plot, x='xpos', y = 'ypos', hue = stim_group, palette=colormap, s=200,ax=ax[0][2])
    ax[0][2].set_title('Stimulation Postions')
    ax[0][2].axis('off')


    pp = np.array(([0,1], [1,0], [1,1], [1,2]))
    for ii, sensor in enumerate(chansplot):

        dat_plot = eegdat_df.loc[eegdat_df.ch_name==sensor]
        dat_plot = dat_plot.loc[eegdat_df[stim_group]!=0]
        dat_plot = dat_plot.loc[:,['sub_id', 'time (s)', stim_group, 'EEG amp. (µV)']].groupby(by=['sub_id', 'time (s)', stim_group]).mean()

        axuse = ax[pp[ii][0]][pp[ii][1]]
        axuse.axvline(x=0, color='k')
        axuse.axhline(y=0, color='k')
        sns.lineplot(data=dat_plot, x='time (s)', y = 'EEG amp. (µV)', hue = stim_group, palette=colormap, ax=axuse)
        axuse.set_title(sensor)
        axuse.spines[['right', 'top']].set_visible(False)

    plt.savefig(direct['resultsroot'] / Path('GroupERPs' + stim_group + '.png'))

# %% Step 2 - Different Horizontal and vertical groups topos

# Plot Topos for the two timed positions
for stim_group in [ 'vert_group', 'horz_group']: #'vert_group'
    if stim_group == 'horz_group':
        timepoints = [0.175] # horz
        colormap = "viridis"

    if stim_group == 'vert_group':
        timepoints = [0.18] # vert
        colormap = 'viridis'

    # get corrected timepoints
    timepoints_use = []
    for tt, time in enumerate (timepoints):
        tmp = np.argmin(np.abs(epochs.times-time))
        timepoints_use.append(epochs.times[tmp])

    # generate figures
    fig, ax = plt.subplots(1,6, layout='tight', figsize=(16,5))

    for tt, time in enumerate (timepoints_use):
        for gg, group in enumerate([-3, -2, -1, 1, 2, 3]):

            dat_plot = eegdat_df.loc[(eegdat_df['time (s)']> time-0.05) & (eegdat_df['time (s)']< time+0.05)]
            # dat_plot = eegdat_df.loc[eegdat_df['sub_id']== 'sub1']
            dat_plot = dat_plot.loc[eegdat_df[stim_group]==group]
            dat_plot = dat_plot.loc[:,['ch_name', 'EEG amp. (µV)']].groupby(by=['ch_name']).mean()

            axuse = ax[gg]# ax[tt,gg]

            im=mne.viz.plot_topomap(np.squeeze(dat_plot), epochs.info, cmap=colormap, vlim=[-1,1], sphere= 'eeglab', contours=0, image_interp='cubic', axes=axuse)

            tit = 'Group:' + str(group) + ', Time:' + str(np.round(time*100)/100)
            axuse.set_title(tit)

    plt.suptitle('GroupTopos' + stim_group)
    plt.savefig(direct['resultsroot'] / Path('GroupTopos' + stim_group + '.png'))
#RdYlGn, RdYlBu,  Spectral, coolwarm, RdBu, PRGn, PuOr,

# %% organise data for PCA
dat_use = eegdat_df.loc[eegdat_df['time (s)']>0,:]
dat_use = dat_use .loc[:,['ch_name', 'time (s)', 'site_id', 'EEG amp. (µV)']].groupby(by=['ch_name', 'time (s)', 'site_id']).mean() # average across all subjects

datuse_export = dat_use.reset_index().pivot(columns='site_id', index = ['ch_name', 'time (s)'], values= 'EEG amp. (µV)')

from scipy.stats import zscore
datuse_export = zscore(datuse_export)
datuse_export = datuse_export.to_numpy()


# %% Trying things
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
X= datuse_export
pca.fit(X.T)
X_pca = pca.fit_transform(X.T)
X_pca.shape

# %matplotlib qt

fig, ax = plt.subplots(1,3,subplot_kw={'projection':'3d'}, figsize=(10,5))
ax[0].scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2], c=stim_df.xpos, cmap='coolwarm', s=30)
ax[0].set_title('X position')

ax[1].scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2], c=stim_df.ypos, cmap='coolwarm', s=30)
ax[1].set_title('Y position')

ax[2].scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2], c=stim_df.eccintricity, cmap='coolwarm', s=30)
ax[2].set_title('Eccintricity')

plt.suptitle('PCA components')
plt.savefig(direct['resultsroot'] / Path('PCA components.png'))

# %% Prepare data for PLSC

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

# %% # Normalise
# Both X and Y are centered and normalized within each "condition" (subject)
# n (i.e., each Xn and Yn is centered and normalized independently,
# and the sum of squares of a column in one condition is equal to 1,
X_norm, Y_norm = [],[]
for sub in range(nsubs):
    print('Normalising data for subject: ' + str(sub))

    # get data
    x_n, y_n = np.squeeze(X[sub]), np.squeeze(Y[sub])

    ## Zero mean each column
    [x_n, y_n] = [dat - np.tile(dat.mean(axis=0).T, (n_sites, 1)) for dat in [x_n, y_n]]

    # divide by sqrt of sum of squares
    [x_ss, y_ss] = [np.sqrt(np.sum(np.square(dat), axis=0)) for dat in [x_n, y_n]]
    [x_n, y_n] = [dat/np.tile(ss.T, (n_sites, 1)) for dat, ss in zip([x_n, y_n], [x_ss, y_ss])]

    # Store
    X_norm.append(x_n)
    Y_norm.append(y_n)

# %% compute R (covariance)

R=[]
for sub in range(nsubs):
    print('Computing covariance for subject: ' + str(sub))

    # get data
    x_n, y_n = X_norm[sub], Y_norm[sub]

    # compute covariance
    R.append(np.dot(y_n.T, x_n))

# %% Restructure variables
# concatenate into a single matrix
R = np.concatenate(R, axis=0)
X = np.squeeze(np.concatenate(X_norm, axis=0))
Y = np.squeeze(np.concatenate(Y_norm, axis=0))

# %% compute SVD

U,D,V = np.linalg.svd(R, full_matrices = False)
V = V.T
print("U matrix size :", U.shape, "D matrix size :", D.shape, "V matrix size :", V.shape)


# %% Visualise behavioural saliances

U_df = pd.DataFrame(U)
U_df['Measure'] = [behave for sub in range(nsubs)for behave in str_behave]
U_df['SUB'] = [sub for sub in range(nsubs) for behave in str_behave]


if nsubs>1:
    plt.figure()
    sns.barplot(data=U_df, x='Measure', y=0, hue='SUB')
else:
    fig, ax = plt.subplots(1, len(U_df), layout = 'tight', figsize = (12,4), sharey= True)
    for component in range(len(U_df)):
        sns.barplot(data=U_df, x='Measure', y=component, ax=ax[component])
        ax[component].set_title('component: ' + str(component))

#%% Compute latent variables

Lx = np.matmul(X, V)
Ly = []
for sub in range(nsubs):
    idx = U_df['SUB'] == sub
    Ly.append(np.matmul(Y_norm[sub], U[idx, :]))
Ly = np.squeeze(np.concatenate(Ly, axis=1))

np.savez(direct['resultsroot'] / Path('SVDcomponents.npz'), U=U, D=D,V=V, Lx=Lx, stim_df=stim_df)
# %% Visualise latent variables

# create new colour map
cols = []
for site in stim_df.index:
    # set colours
    if (stim_df.angle[site] > 0) & (stim_df.angle[site] < 90): col = plt.colormaps['Purples']((stim_df.eccintricity[site]+1)/7)
    if (stim_df.angle[site] > 90) &  (stim_df.angle[site] < 180): col = plt.colormaps['Blues']((stim_df.eccintricity[site]+1)/7)
    if (stim_df.angle[site] > 180) &  (stim_df.angle[site] < 270):col = plt.colormaps['YlOrBr']((stim_df.eccintricity[site]+1)/7)
    if (stim_df.angle[site] > 270) &  (stim_df.angle[site] < 360): col = plt.colormaps['Reds']((stim_df.eccintricity[site]+1)/7)
    cols.append(col)
from matplotlib.colors import ListedColormap
newcmp = ListedColormap(cols)
# set index
subidx = np.arange(0,60)

# plot
fig, ax = plt.subplots(1,3, figsize=(10,4), layout='tight')
ax[0].scatter(Lx[subidx, 1],Lx[subidx, 0],c=stim_df.eccintricity, cmap="coolwarm", s=50)
ax[0].set_title('eccintricity')

ax[1].scatter(Lx[subidx, 1],Lx[subidx, 0],c=stim_df.angle, cmap='coolwarm', s=50)
ax[1].set_title('angle')

plt.scatter(Lx[subidx, 1],Lx[subidx, 0],c=stim_df.site_number, cmap=newcmp, s=50)
ax[2].set_title('stimid')

[ax[a].set_xlabel('Latent X1') for a in range(3)]
[ax[a].set_ylabel('Latent X2') for a in range(3)]


plt.suptitle('Latent variables X')
plt.savefig(direct['resultsroot'] / Path('PLSC Latent variables.png'))

# %% Visualise latent variables
subidx = np.arange(0,60)
# subidx = np.arange(60,120)
# subidx = np.arange(120,180)
fig, ax = plt.subplots(1,4,subplot_kw={'projection':'3d'}, figsize=(15,5))
ax[0].scatter(Lx[subidx, 0],Lx[subidx, 1],Lx[subidx, 2], c=stim_df.xpos, cmap='coolwarm', s=50)
ax[0].set_title('xpos')

ax[1].scatter(Lx[subidx, 0],Lx[subidx, 1],Lx[subidx, 2], c=stim_df.ypos, cmap='coolwarm', s=50)
ax[1].set_title('ypos')

ax[2].scatter(Lx[subidx, 0],Lx[subidx, 1],Lx[subidx, 2], c=stim_df.eccintricity, cmap='coolwarm', s=50)
ax[2].set_title('Eccintricity')

ax[3].scatter(Lx[subidx, 0],Lx[subidx, 1],Lx[subidx, 2], c=stim_df.site_number, cmap=newcmp, s=50)
ax[3].set_title('SiteID')

plt.suptitle('Latent variables X')
plt.savefig(direct['resultsroot'] / Path('PLSC Latent variables 3d.png'))
