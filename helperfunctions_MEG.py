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
    behvars = ['cond_id', 'stimsize', 'x_pos', 'y_pos', 'angle', 'x_dir', 'ydir']

    def get_subids(self):
        subids = os.listdir(self.direct['dataroot'] / Path('MEG'))
        return subids

    def get_eeg(self):  # Load and store EEG data
        subids = self.get_subids()

        # Preallocate
        megdat_dict = {'sub_id': [], 'epoch': [], 'rep': [], 'sensor': [], 'Amplitude (T)': [], 'cond_id': [],
                       'stimsize': [], 'x_pos': [], 'y_pos': [], 'angle': [], 'x_dir': [], 'y_dir': []}

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
            for rep in range(2):
                for sensor in range(self.n_sensors):
                    for epoch in epochs_use:
                        megdat_dict['sub_id'].append(subid)
                        megdat_dict['epoch'].append(epoch)
                        megdat_dict['rep'].append(rep)
                        megdat_dict['sensor'].append(sensor)
                        megdat_dict['Amplitude (T)'].append(MEGdat[sensor, rep, epoch])

                        # condition labels
                        stimuse = STIMdat[epoch, :, :]
                        megdat_dict['cond_id'].append(CONDdat[:, epoch][0])
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
                        if 10 < angle < 80:
                            angle = 45
                        if -10 > angle > -80:
                            angle = 45
                        megdat_dict['angle'].append(angle)

                        # if (rep == 1) & (sensor == 1) & (np.mod(epoch,20)==0):
                        #     plt.figure()
                        #     plt.imshow(stimuse.T)
                        #     plt.title('Epoch = ' + str(epoch) + '| Lengths x,y = ' + str(lengths['x']) + ',' + str(lengths['y']) + '| Angle = ' + str(angle))

        # calculate motion directions
        xdiff, ydiff = np.diff(megdat_dict['x_pos'], prepend=0), np.diff(megdat_dict['y_pos'], prepend=0) # works because we cycle through epochs in the inner loop and epochs are sequential in time
        xdiff[np.abs(xdiff) > 6],  ydiff[np.abs(ydiff) > 6] = 0, 0
        xdiff, ydiff = np.sign(xdiff), np.sign(ydiff)
        megdat_dict['x_dir'], megdat_dict['y_dir'] = xdiff, ydiff

        # plt.figure()
        # plt.plot(megdat_dict['x_pos'])
        # plt.plot(megdat_dict['y_pos'])
        #
        # plt.figure()
        # plt.plot(np.sign(xdiff))
        # plt.plot(np.sign(ydiff))

        megdat_df = pd.DataFrame(megdat_dict)

        for key in megdat_dict.keys():
            print(key + str(len(megdat_dict[key])))
        # save data
        megdat_df.to_pickle(self.files['MEGprocessed'])

        return megdat_df

    # Prepare data for PLSC
    def organise_megdata(self, megdat_df, grouper=True):

        # Get copy of data
        meg_resampled = megdat_df.copy()
        meg_resampled = meg_resampled.dropna(axis=0)

        # group
        if grouper:
            meg_resampled = meg_resampled.groupby(['cond_id', 'epoch', 'sensor', 'rep'])[['stimsize', 'x_pos', 'y_pos', 'angle', 'x_dir', 'y_dir', 'Amplitude (T)']].mean().reset_index()
            meg_resampled['sub_id'] = 'sub1'
            nsubs = 1

        else:
            nsubs = self.n_subs

        return meg_resampled, nsubs


# define PLSC class
class PLSC:
    def stack_data(self, nsubs, str_behaveuse, meg_resampled):
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
            for epoch in meg_resampled.epoch.unique():
                # get condition labels
                ydat = meg_resampled.loc[(meg_resampled['sub_id'] == ('sub' + str(sub+1))) & (meg_resampled.epoch == epoch), str_behaveuse].mean().to_numpy().T
                xdat = meg_resampled.loc[(meg_resampled['sub_id'] == ('sub' + str(sub+1))) & (meg_resampled.epoch == epoch), ['Amplitude (T)']]
                # question - make sure it's always the same order of sensors etc.

                # resample EEG data
                x_n.append(xdat.to_numpy())
                y_n.append(ydat)

            X_stack.append(np.stack(x_n))
            Y_stack.append(np.stack(y_n))

        return X_stack, Y_stack

    def normalise(self, X_stack, Y_stack, nsubs, sets):
        X_norm, Y_norm = [],[]
        normstats = {'X_mean':[], 'X_ss': []}
        for sub in range(nsubs):
            print('Normalising data for subject: ' + str(sub))

            # get data
            x_n, y_n = np.squeeze(X_stack[sub]), np.squeeze(Y_stack[sub])
            luse = x_n.shape[0]
            ## Zero mean each column
            [x_n, y_n] = [dat - np.tile(dat.mean(axis=0).T, (luse, 1)) for dat in [x_n, y_n]]
            normstats['X_mean'].append(x_n.mean(axis=0))

            # divide by sqrt of sum of squares
            [x_ss, y_ss] = [np.sqrt(np.sum(np.square(dat), axis=0)) for dat in [x_n, y_n]]
            [x_n, y_n] = [dat/np.tile(ss.T, (luse, 1)) for dat, ss in zip([x_n, y_n], [x_ss, y_ss])]
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

    def compute_SVD(self, R, X_norm, Y_norm, sets, str_behaveuse, nsubs):
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
        U_df['Measure'] = [behave for sub in range(nsubs)for behave in str_behaveuse]
        U_df['SUB'] = [sub for sub in range(nsubs) for behave in str_behaveuse]

        return U, D, V, X, Y, U_df

    def compute_latentvars(self, sets, X, Y_norm, U_df, U, V, D, nsubs):
        Lx = np.matmul(X, V)
        Ly = []
        for sub in range(nsubs):
            idx = U_df['SUB'] == sub
            Ly.append(np.matmul(Y_norm[sub], U[idx, :]))
        Ly = np.squeeze(np.concatenate(Ly, axis=1))

        return Lx, Ly
