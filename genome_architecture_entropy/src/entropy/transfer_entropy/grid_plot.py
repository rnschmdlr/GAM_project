# %%
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as lines
import seaborn as sns
import itertools

grid = np.load('/Users/pita/Documents/Rene/GAM_project/genome_architecture_entropy/data/results/grid_search.npy')
series_lengths = [5, 10, 20, 40]
num_params = [1, 5, 10, 20]
self_dep = [0.0, 0.2, 0.4, 0.8]

sns.set_theme(style="white")

#%%
# Set up the matplotlib figure
f, axes = plt.subplots(4, 4, figsize=(8, 8))

#axes.set
# Rotate the starting point around the cubehelix hue circle
#for ax, s in zip(axes.flat, np.linspace(0, 3, 10)):
for ax, (x, y) in zip(axes.flat, list(itertools.product(reversed(range(4)), range(4)))):
    slice = grid[x,y]
    #slice[slice < 0.7] = 0 # tresholding

    series_len = series_lengths[x]
    num_param = num_params[y]
    history_length = [1, 2, 4, series_len]

    cmap = sns.color_palette('flare', as_cmap=True)
    sns.heatmap(
        data=slice,
        cmap=cmap,
        ax=ax,
        cbar=False,
        vmax=1,
        vmin=0,
        square=True,
        linewidths=0)
    
    ax.tick_params(axis=u'both', which=u'both',length=0, pad=5)
    ax.tick_params(labelleft=False,
                    labelbottom=False,
                    labeltop=False,
                    labelright=False)

    if x == 3:
        ax.set(xticklabels=self_dep)
        ax.xaxis.set_ticks_position('top')
        
    if x == 0:
        ax.set_xlabel(num_param, labelpad=24)
        ax.set(xticklabels=[])

    if y == 3:
        ax.set(yticklabels=history_length)
        ax.yaxis.set_ticks_position('right')
        
    if y == 0:
        ax.set_ylabel(series_len, labelpad=22)
        ax.set(yticklabels=[])
        ax.yaxis.set_label_position('left')

    if x == 3 and y == 3:
        ax.set_xlabel('self-dependence', labelpad=10, size=13)
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('length of history', labelpad=10, size=13)
        ax.yaxis.set_label_position('right')

f.subplots_adjust(0.13, 0.1, 1, 1, 0.15, 0.15)

f.supxlabel('number of parameters', x=0.57)
f.supylabel('length of series', y=0.55)

f.add_artist(lines.Line2D([0.1, 0.1], [0.105, 0.995], color='black')) # left line
f.add_artist(lines.Line2D([0.13, 1], [0.07, 0.07], color='black')) # bottom line