import numpy as np
import pandas as pd
from pathlib import Path
import mne


class MetaData:

    # %% Metadata
    n_sites = 60
    n_electrodes = 59
    n_subs = 28

    # %% Directory structure
    direct = {'dataroot': Path('Data/'), 'resultsroot': Path('Results/')}
    files = {'EEGprocessed': direct['dataroot'] / 'EEGdataframe.pkl'}

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


    # Organise some plotting groups we might want to plot by
    horz_groups = {'-3': [57, 58, 45, 46], '-2': [34, 22, 33, 21], '-1': [11, 4, 10, 3],
                   '1': [1, 6, 2, 7], '2': [15, 27, 16, 28], '3': [39, 51, 40, 52]}

    vert_groups = {'-3': [55, 54, 43, 42], '-2': [31, 30, 19, 18], '-1': [9, 8, 3, 2],
                   '1': [1, 4, 5, 12], '2': [24, 13, 36, 25], '3': [48, 37, 60, 49]}

    # assign horizontal groups
    stim_df['horz_group'] = 0
    for group in horz_groups:
        for index in horz_groups[group]:
            stim_df.loc[stim_df.site_number == index, 'horz_group'] = int(group)

    # assign vertical groups
    stim_df['vert_group'] = 0
    for group in vert_groups:
        for index in vert_groups[group]:
            stim_df.loc[stim_df.site_number == index, 'vert_group'] = int(group)


    def get_eeginfo(self):  # Generate EEG info structure
        # load raw data
        fname = self.direct['dataroot'] / Path("Datasample/trialtest.mat")
        epochs = mne.read_epochs_fieldtrip(fname, info=None, data_name='tmp',trialinfo_column=0)

        # Create empty info structure
        info = mne.create_info(epochs.info['ch_names'], epochs.info['sfreq'], ch_types='eeg')

        # Set 10-20 montage
        montage2use = mne.channels.make_standard_montage('standard_1020')  # mne.channels.get_builtin_montages()
        info = info.set_montage(montage2use)

        return info

    def get_eeg(self):  # Load and store EEG data
        # Preallocate
        eegdat_dict = {'sub_id': [], 'site_id': [], 'ch_name': [], 'time (s)': [], 'EEG amp. (µV)': []}
        info = self.get_eeginfo()

        # loop through subjects
        for sub in range(self.n_subs):
            self.direct['datasub'] = self.direct['dataroot'] / Path('S' + str(sub+1) + '/')

            # loop through sites
            for site in range(self.n_sites):
                # load data
                fname = self.direct['datasub'] / Path("data_s" + str(sub + 1) + '_site' + str(site + 1) + ".mat")
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
        grouper = True
        if grouper:
            eeg_resampled = eeg_resampled.groupby(['site_id', 'ch_name', 'time (s)'])['EEG amp. (µV)'].mean().reset_index()
            eeg_resampled['sub_id'] = 'sub1'
            nsubs = 1
        else:
            nsubs = 28

        # save data
        eeg_resampled.to_pickle(self.direct['resultsroot'] / Path('eegretinotopicmappingdf.pkl'))

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
        for sub in range(nsubs):
            print('Normalising data for subject: ' + str(sub))

            # get data
            x_n, y_n = np.squeeze(X[sub]), np.squeeze(Y[sub])

            ## Zero mean each column
            [x_n, y_n] = [dat - np.tile(dat.mean(axis=0).T, (sets.n_sites, 1)) for dat in [x_n, y_n]]

            # divide by sqrt of sum of squares
            [x_ss, y_ss] = [np.sqrt(np.sum(np.square(dat), axis=0)) for dat in [x_n, y_n]]
            [x_n, y_n] = [dat/np.tile(ss.T, (sets.n_sites, 1)) for dat, ss in zip([x_n, y_n], [x_ss, y_ss])]

            # Store
            X_norm.append(x_n)
            Y_norm.append(y_n)

        return X_norm, Y_norm

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
