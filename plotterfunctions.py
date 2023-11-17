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
    # Set 10-20 montage
    montage2use = mne.channels.make_standard_montage('standard_1020')  # mne.channels.get_builtin_montages()
    info= info.set_montage(montage2use)
    fig = info.plot_sensors(show_names=True, block=False, sphere="eeglab")
