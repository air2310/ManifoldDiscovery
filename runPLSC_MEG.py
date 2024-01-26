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
run_fullgroup = True

# leave one out cross validation
run_crossvalidation = False

# %% Get metadata
sets = helper.MetaData()
# info = sets.get_eeginfo()
plsc = helper.PLSC()
subs = sets.get_subids()

# %% Load MEG data

if loadMEGfresh:
    megdat_df = sets.get_eeg()
else:
    megdat_df = pd.read_pickle(sets.files['MEGprocessed'])

if run_fullgroup:
    # set behavioural variables
    str_behaveuse = [ 'stimsize', 'x_pos', 'y_pos', 'angle', 'x_dir', 'y_dir']

    # get resampled variables
    meg_resampled, nsubs = sets.organise_megdata(megdat_df, grouper=True) # grouper = average data across participants?

    # %% stack data up across subjects and conditions to form X and Y
    # X = [[cond, ch x times]Pid, [cond, ch x times]Pid ..., [cond, ch x times]pidn] .
    # Y = [[cond,[behave values i.e. xpos, ypos]]p1, ..
    plsc = helper.PLSC()
    X_stack, Y_stack = plsc.stack_data(nsubs, str_behaveuse, meg_resampled)

    # %% # Normalise
    # Both X and Y are centered and normalized within each "condition" (subject)
    # n (i.e., each Xn and Yn is centered and normalized independently,
    # and the sum of squares of a column in one condition is equal to 1,
    X_norm, Y_norm, normstats = plsc.normalise(X_stack, Y_stack, nsubs, sets)

    # %% compute R (covariance)
    R = plsc.compute_covariance(X_norm, Y_norm, sets, nsubs)

    # %% Restructure variables
    U, D, V, X, Y, U_df = plsc.compute_SVD( R, X_norm, Y_norm,sets, str_behaveuse, nsubs)

    #%% Compute latent variables
    Lx, Ly = plsc.compute_latentvars(sets, X, Y_norm, U_df, U, V, D, nsubs)
    # np.savez(sets.direct['resultsroot'] / Path('SVDcomponents.npz'), U=U, D=D,V=V, Lx=Lx, stim_df=setes.stim_df)

    # %% Visualise latent vars
    sets.nsubs = nsubs
    plot.behave_saliances(U_df, sets)
    plot.latentspace_2d(sets, Lx)
    plot.latentspace_3d(sets, Lx)

    str_behaveuse = [ 'stimsize', 'x_pos', 'y_pos', 'angle', 'x_dir', 'y_dir']
    fig, ax = plt.subplots(1,3, figsize=(10,4), layout='tight')
    ax[0].scatter(Lx[:, 1],Lx[:, 0],c=Y[:,0],cmap="coolwarm", s=50)
    ax[1].scatter(Lx[:, 1],Lx[:, 0],c=Y[:,1],cmap="coolwarm", s=50)
    ax[2].scatter(Lx[:, 1],Lx[:, 2],c=Y[:,2],cmap="coolwarm", s=50)

    fig, ax = plt.subplots(1,3, figsize=(10,4), layout='tight')
    ax[0].scatter(Lx[:, 4],Lx[:, 3],c=Y[:,5],cmap="coolwarm", s=50)
    ax[1].scatter(Lx[:, 1],Lx[:, 4],c=Y[:,4],cmap="coolwarm", s=50)
    ax[2].scatter(Lx[:, 3],Lx[:, 5],c=Y[:,3],cmap="coolwarm", s=50)

    # plot reconstructed latent space vars
    eeg_V = plot.latentspaceERPs(sets, eeg_resampled, V)
    plot.animatecomponents(eeg_V, info, sets, component='V0')
