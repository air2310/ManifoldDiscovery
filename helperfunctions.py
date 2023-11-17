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
