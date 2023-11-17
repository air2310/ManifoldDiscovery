import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import mne

def radialcoords(sets):
    stim_df = sets.stim_df

    # Plot result
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
    for site in stim_df.index:
        # set colours
        if (stim_df.angle[site] > 0) & (stim_df.angle[site] < 90): col = plt.colormaps['Purples']((stim_df.eccintricity[site]+1)/7)
        if (stim_df.angle[site] > 90) & (stim_df.angle[site] < 180): col = plt.colormaps['Blues']((stim_df.eccintricity[site]+1)/7)
        if (stim_df.angle[site] > 180) & (stim_df.angle[site] < 270):col = plt.colormaps['YlOrBr']((stim_df.eccintricity[site]+1)/7)
        if (stim_df.angle[site] > 270) & (stim_df.angle[site] < 360): col = plt.colormaps['Reds']((stim_df.eccintricity[site]+1)/7)
        ax.plot(np.deg2rad(stim_df.angle[site]), stim_df.eccintricity[site]+1, marker='o', markerfacecolor = col, markeredgecolor = col, markersize=20, alpha=0.75)
        plt.text(np.deg2rad(stim_df.angle[site]), stim_df.eccintricity[site]+1, str(site+1), color=[0.4,0.4,0.4], horizontalalignment='center',
                 verticalalignment='center')

    ax.set_title("Stimulation positions", va='bottom')
    ax.set_rlabel_position(90)  # Move radial labels away from plotted line
    ax.set_rlim(0,7)
    ax.grid(True)
    plt.savefig(sets.direct['resultsroot'] / Path('RadialCoords.png'))


def montage(info):
    fig = info.plot_sensors(show_names=True, block=False, sphere="eeglab")


def ERPs(sets, eegdat_df, info):
    stim_df = sets.stim_df

    # loop though stimulus groups
    for stim_group in [ 'horz_group','vert_group']:
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
        dat_plot = stim_df.loc[stim_df[stim_group] == 0]
        sns.scatterplot(data=dat_plot, x='xpos', y = 'ypos', color='grey', s=100, ax=ax[0][2])
        dat_plot = stim_df.loc[stim_df[stim_group] != 0]
        sns.scatterplot(data=dat_plot, x='xpos', y = 'ypos', hue = stim_group, palette=colormap, s=200,ax=ax[0][2])
        ax[0][2].set_title('Stimulation Postions')
        ax[0][2].axis('off')

        pp = np.array(([0, 1], [1, 0], [1, 1], [1, 2]))
        for ii, sensor in enumerate(chansplot):

            dat_plot = eegdat_df.loc[eegdat_df.ch_name == sensor]
            dat_plot = dat_plot.loc[eegdat_df[stim_group]!=0]
            dat_plot = dat_plot.loc[:,['sub_id', 'time (s)', stim_group, 'EEG amp. (µV)']].groupby(by=['sub_id', 'time (s)', stim_group]).mean()

            axuse = ax[pp[ii][0]][pp[ii][1]]
            axuse.axvline(x=0, color='k')
            axuse.axhline(y=0, color='k')
            sns.lineplot(data=dat_plot, x='time (s)', y = 'EEG amp. (µV)', hue = stim_group, palette=colormap, ax=axuse)
            axuse.set_title(sensor)
            axuse.spines[['right', 'top']].set_visible(False)

        plt.savefig(sets.direct['resultsroot'] / Path('GroupERPs' + stim_group + '.png'))


def topos(sets, eegdat_df, info):

    # Plot Topos for the two timed positions
    for stim_group in [ 'vert_group', 'horz_group']: #'vert_group'
        if stim_group == 'horz_group':
            timepoints = [0.175] # horz
            colormap = "viridis"

        if stim_group == 'vert_group':
            timepoints = [0.18] # vert
            colormap = 'viridis'

        # get corrected timepoints
        times = eegdat_df['time (s)'].groupby('time (s)').mean().reset_index()

        timepoints_use = []
        for tt, time in enumerate (timepoints):
            tmp = np.argmin(np.abs(times-time))
            timepoints_use.append(times[tmp])

        # generate figures
        fig, ax = plt.subplots(1,6, layout='tight', figsize=(16,5))

        for tt, time in enumerate (timepoints_use):
            for gg, group in enumerate([-3, -2, -1, 1, 2, 3]):

                dat_plot = eegdat_df.loc[(eegdat_df['time (s)'] > time-0.05) & (eegdat_df['time (s)']< time+0.05)]
                # dat_plot = eegdat_df.loc[eegdat_df['sub_id']== 'sub1']
                dat_plot = dat_plot.loc[eegdat_df[stim_group] == group]
                dat_plot = dat_plot.loc[:,['ch_name', 'EEG amp. (µV)']].groupby(by=['ch_name']).mean()

                axuse = ax[gg]# ax[tt,gg]

                im=mne.viz.plot_topomap(np.squeeze(dat_plot), info, cmap=colormap, vlim=[-1,1], sphere= 'eeglab', contours=0, image_interp='cubic', axes=axuse)

                tit = 'Group:' + str(group) + ', Time:' + str(np.round(time*100)/100)
                axuse.set_title(tit)

        plt.suptitle('GroupTopos' + stim_group)
        plt.savefig(sets.direct['resultsroot'] / Path('GroupTopos' + stim_group + '.png'))
    #RdYlGn, RdYlBu,  Spectral, coolwarm, RdBu, PRGn, PuOr,
