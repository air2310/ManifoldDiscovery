# %% import libraries
import helperfunctions as helper
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
loadEEGfresh = True
visualiseEEG = True
loadEEGresampled_fresh = False

# full group
run_fullgroup = False

# leave one out cross validation
run_crossvalidation = False

# %% Get metadata
sets = helper.MetaData()
info = sets.get_eeginfo()
plsc = helper.PLSC()
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

# %% Run cross validation
# set behavioural variables
str_behave = ['eccintricity', 'xpos', 'ypos']  # ['angle', 'visfield_horz', 'visfield_vert']
if run_crossvalidation:

    # %% get resampled variables
    if loadEEGresampled_fresh:
        eeg_resampled_all, nsubs = sets.organise_eegdata(eegdat_df, str_behave, grouper=False) # grouper = average data across participants?
    else:
        eeg_resampled_all, nsubs = pd.read_pickle(sets.direct['resultsroot'] / 'eegretinotopicmappingdf_ungrouped.pkl'), 28

    # %% Loop through excluded subjects

    # classify
    metrics = ['horz_group', 'vert_group', 'eccintricity', 'visfield_vert', 'visfield_horz']
    classification_acc = {metric: [] for metric in metrics}
    classification_acc['sub_id'] = []
    classification_acc['n_epoch_select'] = []
    classification_acc['distance'] = []
    distances = np.empty((sets.n_sites, sets.n_subs))

    for n_epochs_select in [1, 2, 4, 8, 16, 32, 'max']:
        for sub in np.arange(sets.n_subs):
            subexclude = 'sub' + str(sub+1)

            # segment
            leftindat = eeg_resampled_all.loc[~(eeg_resampled_all.sub_id == subexclude)]

            # group remaining data
            eeg_resampled = eeg_resampled_all.groupby(['site_id', 'ch_name', 'time (s)'])['EEG amp. (µV)'].mean().reset_index()
            eeg_resampled['sub_id'] = 'sub1'
            nsubs = 1

            # %% stack data up across subjects and conditions to form X and Y
            # X = [[cond, ch x times]Pid, [cond, ch x times]Pid ..., [cond, ch x times]pidn] .
            # Y = [[cond,[behave values i.e. xpos, ypos]]p1, ..
            X_stack, Y_stack = plsc.stack_data(nsubs, sets, str_behave, eeg_resampled)

            # %% # Normalise
            # Both X and Y are centered and normalized within each "condition" (subject)
            # n (i.e., each Xn and Yn is centered and normalized independently,
            # and the sum of squares of a column in one condition is equal to 1,
            X_norm, Y_norm, normstats = plsc.normalise(X_stack, Y_stack, nsubs, sets)

            # %% compute R (covariance)
            R = plsc.compute_covariance(X_norm, Y_norm, sets, nsubs)

            # %% Restructure variables
            U, D, V, X, Y, U_df = plsc.compute_SVD( R, X_norm, Y_norm,sets, str_behave, nsubs)

            #%% Compute latent variables
            Lx, Ly = plsc.compute_latentvars(sets, X, Y_norm, U_df, U, V, D, nsubs)
            # np.savez(sets.direct['resultsroot'] / Path('SVDcomponents.npz'), U=U, D=D,V=V, Lx=Lx, stim_df=setes.stim_df)

            # %% compute for left out data

            # resample leftoutdat
            eegdat_df_leftout = sets.get_eeg(subs=[sub], n_epochs_select=n_epochs_select)
            leftoutdat, nsubs = sets.organise_eegdata(eegdat_df_leftout, str_behave, grouper=True)
            leftoutdat.loc[:, 'sub_id'] = 'sub1'

            # get stacked left out data as well
            X_leftout, Y_leftout = plsc.stack_data(nsubs, sets, str_behave, leftoutdat)

            # normalise left out data with group stats
            X_normleftout = plsc.normalise_Wnormstats(X_leftout, sets, normstats)

            # Compute matching latent X using normstats for left out person
            Lx_lo, Ly = plsc.compute_latentvars(sets,  X_normleftout, Y_norm, U_df, U, V, D, nsubs)

            # %% Score (distance metrics and acc)
            # score distance
            score = Lx - Lx_lo
            distance = np.sqrt(np.sum(np.square(score),1))
            distances[:, sub] = distance

            #classify
            classification_acc['sub_id'].append(subexclude)
            classification_acc['n_epoch_select'].append(n_epochs_select)
            classification_acc['distance'].append(np.mean(distance))

            for metric in metrics:
                neigh = KNeighborsClassifier(n_neighbors=4)
                neigh.fit(Lx, sets.stim_df[metric])
                classification_acc[metric].append(neigh.score(Lx_lo, sets.stim_df[metric]))

            # classify noise
            if n_epochs_select == 'max':
                classification_acc['sub_id'].append(subexclude)
                classification_acc['n_epoch_select'].append('randombaseline')

                for metric in metrics:
                    neigh = KNeighborsClassifier(n_neighbors=4)
                    neigh.fit(Lx, sets.stim_df[metric])

                    score = []
                    for shuff in range(30):
                        np.random.shuffle(Lx_lo)
                        score.append(neigh.score(Lx_lo, sets.stim_df[metric]))

                    classification_acc[metric].append(np.mean(score))

                score = Lx - Lx_lo
                classification_acc['distance'].append(np.mean(np.sum(np.square(score),1)))

            # Visualise latent vars
            sets.nsubs = nsubs
            plot.latentspace_2dLO(sets, Lx,Lx_lo)

    for key in classification_acc:
        print(key + str(len(classification_acc[key])))

    # save classification acc
    classif_pd = pd.DataFrame(classification_acc)
    classif_pd.to_pickle(sets.direct['resultsroot'] / 'crossvalacc.pkl')

else:
    classif_pd = pd.read_pickle(sets.direct['resultsroot'] / 'crossvalacc.pkl')



# %% examine group data

plot.crossvalidation_metrics(classif_pd, sets)

# %% Prepare data for plsc
if run_fullgroup:
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
    X_norm, Y_norm, normstats = plsc.normalise(X_stack, Y_stack, nsubs, sets)

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
    plot.latentspace_2d(sets, Lx)
    plot.latentspace_3d(sets, Lx)

    # plot reconstructed latent space vars
    eeg_V = plot.latentspaceERPs(sets, eeg_resampled, V)
    plot.animatecomponents(eeg_V, info, sets, component='V0')

#%%
