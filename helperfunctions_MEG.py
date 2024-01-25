import numpy as np
import pandas as pd
from pathlib import Path
import mne
import os
import h5py
import math
import matplotlib.pyplot as plt

class MetaData:

    # %% Metadata
    n_epochs = 140
    n_sensors = 157
    n_subs = 30

    # %% Directory structure
    direct = {'dataroot': Path('C:/Users/angel/Documents/MEGRetinotopicMappingData/'), 'resultsroot': Path('Results/')}
    direct['datapreproc'] = direct['dataroot'] / Path('PythonData/')
    files = {'MEGprocessed': direct['dataroot'] / 'MEGdataframe.pkl'}

    # # %% Feature coding for each stimulation location
    # # Create dictionary to host information
    # stim_dict = {'site_number': [], 'eccintricity': [], 'angle': []}
    #
    # # loop through sites to assign properties
    # eccin = 0
    # for site in range(n_sites):
    #     # set angle
    #     if eccin == 0:
    #         angle = np.mod(45 - 90*site, 360)
    #     if eccin == 1:
    #         angle = np.mod(67.5 - 45*(site-4), 360)
    #     if eccin > 1:
    #         angle = np.mod(75 - 30*np.mod(site, 12), 360)
    #
    #     # set properties
    #     stim_dict['site_number'].append(site+1)  # site number
    #     stim_dict['angle'].append(angle)  # site number
    #     stim_dict['eccintricity'].append(eccin)  # eccintricity
    #
    #     # advance eccintricities
    #     if np.isin(site+1, [4, 12, 24, 36, 48, 60]):
    #         eccin += 1
    #
    # # set visual fields
    # stim_dict['visfield_horz'] = (np.mod(stim_dict['angle'], 275) < 90).astype(int)
    # stim_dict['visfield_horz_str'] = np.array(['Left', 'Right'])[stim_dict['visfield_horz']]
    #
    # stim_dict['visfield_vert'] = (np.array(stim_dict['angle']) < 180).astype(int)
    # stim_dict['visfield_vert_str'] = np.array(['Bottom', 'Top'])[stim_dict['visfield_horz']]
    #
    # # get cardinal coordinates
    # stim_dict['xpos'] = (np.array(stim_dict['eccintricity'] ) + 1) * np.cos(np.deg2rad(stim_dict['angle']))
    # stim_dict['ypos'] = (np.array (stim_dict['eccintricity']) + 1) * np.sin(np.deg2rad(stim_dict['angle']))
    #
    # # convert to pandas dataframe
    # stim_df = pd.DataFrame(stim_dict)
    #
    #
    # # Organise some plotting groups we might want to plot by
    # horz_groups = {'-3': [57, 58, 45, 46], '-2': [34, 22, 33, 21], '-1': [11, 4, 10, 3],
    #                '1': [1, 6, 2, 7], '2': [15, 27, 16, 28], '3': [39, 51, 40, 52]}
    #
    # vert_groups = {'-3': [55, 54, 43, 42], '-2': [31, 30, 19, 18], '-1': [9, 8, 3, 2],
    #                '1': [1, 4, 5, 12], '2': [24, 13, 36, 25], '3': [48, 37, 60, 49]}
    #
    # # assign horizontal groups
    # stim_df['horz_group'] = 0
    # for group in horz_groups:
    #     for index in horz_groups[group]:
    #         stim_df.loc[stim_df.site_number == index, 'horz_group'] = int(group)
    #
    # # assign vertical groups
    # stim_df['vert_group'] = 0
    # for group in vert_groups:
    #     for index in vert_groups[group]:
    #         stim_df.loc[stim_df.site_number == index, 'vert_group'] = int(group)

    def get_subids(self):
        subids = os.listdir(self.direct['dataroot'] / Path('MEG'))
        return subids

    def get_eeg(self):  # Load and store EEG data
        subids = self.get_subids()

        # Preallocate
        megdat_dict = {'sub_id': [], 'epoch': [], 'rep': [], 'sensor': [], 'Amplitude (T)': [], 'cond_id': [],
                       'stimsize': [], 'x_pos': [], 'y_pos': [], 'angle': []}

        # loop through subjects
        for ii_sub, subid in enumerate(subids):
            fname = self.direct['datapreproc'] / Path(subid + '.mat')
            fname_stim = self.direct['datapreproc'] / Path(subid + 'stim.mat')
            fname_cond = self.direct['datapreproc'] / Path(subid + 'cond.mat')

            # load data
            f = h5py.File(fname)
            MEGdat = np.array(f['phaseRefMEGResponse'])

            f = h5py.File(fname_stim)
            STIMdat = np.array(f['stimuse'])

            f = h5py.File(fname_cond)
            CONDdat = np.array(f['conds'])
            epochs_use = np.where(np.logical_not(np.isin(CONDdat, [10, 20])))[1]

            # loop through sites
            for epoch in epochs_use:
                for rep in range(2):
                    for sensor in range(self.n_sensors):
                        megdat_dict['epoch'].append(epoch)
                        megdat_dict['rep'].append(rep)
                        megdat_dict['sensor'].append(sensor)
                        megdat_dict['Amplitude (T)'].append(MEGdat[sensor, rep, epoch])

                        # condition labels
                        stimuse = STIMdat[epoch, :, :]
                        megdat_dict['cond_id'].append(CONDdat[:, epoch])
                        megdat_dict['stimsize'].append(stimuse.sum())
                        megdat_dict['x_pos'].append(np.where(stimuse)[0].mean())
                        megdat_dict['y_pos'].append(np.where(stimuse)[1].mean())

                        # calculate angles
                        loc = np.where(stimuse)
                        l_idx = {'x': 0, 'y': 1, '0': 'x', '1': 'y'}
                        span = {ii: [loc[l_idx[ii]].min(), loc[l_idx[ii]].max()] for ii in ['x', 'y']}
                        broader = np.argmax([np.diff(span[ii]) for ii in ['x', 'y']])
                        thinner = int(not broader)
                        idx = [np.argmin(loc[broader]), np.argmax(loc[broader])]
                        lengths = {l_idx[str(broader)]: np.diff(loc[broader][idx])[0],
                                   l_idx[str(thinner)]: np.diff(loc[thinner][idx])[0]}
                        angle = np.rad2deg(math.atan2(lengths['y'], lengths['x']))
                        megdat_dict['angle'].append(angle)

                        plt.figure()
                        plt.imshow(stimuse.T)
                        plt.title('Epoch = ' + str(epoch) + 'lengths x,y = ' + str(lengths['x']) + ',' + str(lengths['y']) + 'angle = ' + str(angle))

            #direction

        megdat_df = pd.DataFrame(megdat_dict)

        # assign to plots to align with our stimulus dataframe
        # assign horizontal groups
        eegdat_df['horz_group'] = 0
        for group in self.horz_groups:
            for index in self.horz_groups[group]:
                eegdat_df.loc[eegdat_df.site_id == index, 'horz_group'] = int(group)

        # assign vertical groups
        eegdat_df['vert_group'] = 0
        for group in self.vert_groups:
            for index in self.vert_groups[group]:
                eegdat_df.loc[eegdat_df.site_id == index, 'vert_group'] = int(group)

        # save data
        eegdat_df.to_pickle(self.files['EEGprocessed'])

        return eegdat_df

    # Prepare data for PLSC
    def organise_eegdata(self, eegdat_df, str_behave, grouper=True):

        # Get positive time values
        eeg_resampled = eegdat_df.copy()
        eeg_resampled = eeg_resampled.loc[eeg_resampled['time (s)'] > 0, :]
        eeg_resampled['time (s)'] = pd.TimedeltaIndex(eeg_resampled['time (s)'],  unit='s')

        # resample
        grouper = eeg_resampled.groupby(['sub_id', 'site_id', 'ch_name'])
        eeg_resampled = grouper.resample('0.01S', on='time (s)', group_keys='True').mean()
        eeg_resampled = eeg_resampled.reset_index()

        # group
        if grouper:
            eeg_resampled = eeg_resampled.groupby(['site_id', 'ch_name', 'time (s)'])['EEG amp. (µV)'].mean().reset_index()
            eeg_resampled['sub_id'] = 'sub1'
            nsubs = 1

            # save data
            eeg_resampled.to_pickle(self.direct['resultsroot'] / Path('eegretinotopicmappingdf.pkl'))
        else:
            nsubs = 28
            # save data
            eeg_resampled.to_pickle(self.direct['resultsroot'] / Path('eegretinotopicmappingdf_ungrouped.pkl'))



        return eeg_resampled, nsubs


# define PLSC class
class PLSC:
    def stack_data(self, nsubs, sets, str_behave, eeg_resampled):
        ## stack data
        # X = [[cond, ch x times]Pid, [cond, ch x times]Pid ..., [cond, ch x times]pidn] .
        # Y = [[xpos, ypos, eccin]p1cond1, ...]

        # preallocate
        X_stack,Y_stack = [], []
        for sub in range(nsubs):
            # preallocate for subject
            print('Getting data for subject: ' + str(sub))
            x_n, y_n = [], []

            # cycle through conditions
            for site in range(sets.n_sites):
                # get condition labels
                ydat = sets.stim_df.loc[sets.stim_df.site_number == (site+1), str_behave].to_numpy()
                xdat = eeg_resampled.loc[(eeg_resampled['sub_id'] == ('sub' + str(sub+1))) & (eeg_resampled.site_id == (site+1)), ['EEG amp. (µV)']]

                # resample EEG data
                x_n.append(xdat.to_numpy())
                y_n.append(ydat)

            X_stack.append(np.stack(x_n))
            Y_stack.append(np.stack(y_n))

        return X_stack, Y_stack

    def normalise(self, X, Y, nsubs, sets):
        X_norm, Y_norm = [],[]
        normstats = {'X_mean':[], 'X_ss': []}
        for sub in range(nsubs):
            print('Normalising data for subject: ' + str(sub))

            # get data
            x_n, y_n = np.squeeze(X[sub]), np.squeeze(Y[sub])

            ## Zero mean each column
            [x_n, y_n] = [dat - np.tile(dat.mean(axis=0).T, (sets.n_sites, 1)) for dat in [x_n, y_n]]
            normstats['X_mean'].append(x_n.mean(axis=0))

            # divide by sqrt of sum of squares
            [x_ss, y_ss] = [np.sqrt(np.sum(np.square(dat), axis=0)) for dat in [x_n, y_n]]
            [x_n, y_n] = [dat/np.tile(ss.T, (sets.n_sites, 1)) for dat, ss in zip([x_n, y_n], [x_ss, y_ss])]
            normstats['X_ss'].append(x_ss)

            # Store
            X_norm.append(x_n)
            Y_norm.append(y_n)


        return X_norm, Y_norm, normstats

    def normalise_Wnormstats(self, X, sets, normstats):

        x_n = np.squeeze(X[0])  # get data
        x_n = x_n - np.tile(normstats['X_mean'][0].T, (sets.n_sites, 1))  ## Zero mean each column
        x_n = x_n / np.tile(normstats['X_ss'][0].T, (sets.n_sites, 1)) # divide by sqrt of sum of squares

        # Store
        X_norm2 = x_n

        return X_norm2

    def compute_covariance(self, X_norm, Y_norm, sets, nsubs):
        R=[]
        for sub in range(nsubs):
            print('Computing covariance for subject: ' + str(sub))

            # get data
            x_n, y_n = X_norm[sub], Y_norm[sub]

            # compute covariance
            R.append(np.dot(y_n.T, x_n))

        return R

    def compute_SVD(self, R, X_norm, Y_norm, sets, str_behave, nsubs):
        # concatenate into a single matrix (from list form)
        R = np.concatenate(R, axis=0)
        X = np.squeeze(np.concatenate(X_norm, axis=0))
        Y = np.squeeze(np.concatenate(Y_norm, axis=0))

        # %% compute SVD

        U,D,V = np.linalg.svd(R, full_matrices = False)
        V = V.T
        print("U matrix size :", U.shape, "D matrix size :", D.shape, "V matrix size :", V.shape)

        # get U dataframe
        U_df = pd.DataFrame(U)
        U_df['Measure'] = [behave for sub in range(nsubs)for behave in str_behave]
        U_df['SUB'] = [sub for sub in range(nsubs) for behave in str_behave]

        return U, D, V, X, Y, U_df

    def compute_latentvars(self, sets, X, Y_norm, U_df, U, V, D, nsubs):
        Lx = np.matmul(X, V)
        Ly = []
        for sub in range(nsubs):
            idx = U_df['SUB'] == sub
            Ly.append(np.matmul(Y_norm[sub], U[idx, :]))
        Ly = np.squeeze(np.concatenate(Ly, axis=1))

        return Lx, Ly
