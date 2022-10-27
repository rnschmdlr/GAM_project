"""
Title: Plot Small Data
Authors: René Schmiedler
Date: 20.08.2021

Description 
-----------
This file allows the user to easily plot data from small data sets and can be imported as a module.

Requires
--------
* numpy
* pandas
* seaborn
* matplotlib

Functions
---------
* plot_xy - a simple plotting helper function for xy coordinates
* heatmap - plots a set of heatmaps from data stored in a dict
* coord   - plots a set of scatterplots from data stored in a dict
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# vispy
# pyviz
# plotly


sns.set_style('darkgrid')        # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes


def plot_xy(mat, xmin, xmax, ymin, ymax, title):
    "This is a simple plotting helper function for xy coordinates."
    sns.set_theme(style="white") #, rc=custom_params)
    plt.show(block=False)
    plt.plot(mat[0], mat[1], '-ok')
    plt.tick_params(
        labelleft=False,
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    plt.axis([xmin-0.5, xmax+0.5, ymin-0.5, ymax+0.5])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal')
    plt.title(title, loc='left')
    plt.show()


def heatmap(dict_arrays, header, v_max):
    """
    This function plots a set of heatmaps from data stored in a dict.
    
    The heatmaps are masked at so that only the lower triangle is drawn.
    
    Arguments
    ---------
    dict_arrays : dict
        arrays stored in a dict to be unpacked
    header : string
        title of the figure
    mask_shape : 
        shape of the mask to be applied to the heatmap
    """

    # Generate a mask for the upper triangle from value shape, k=1 to see diagonal line in heatmap
    mask = np.triu(np.ones((36,36), dtype=bool), k=0)
    f, ax = plt.subplots(3, 2, figsize=(13, 15)) # Set up the matplotlib figure
    cmap = sns.cubehelix_palette(start=5.8, rot=-0.4, dark=0.1, light=0.92, as_cmap=True) # greens
    cmap = sns.cubehelix_palette(start=1, rot=-0.55, dark=0.1, light=0.92, as_cmap=True) # blues
    cmap = sns.color_palette('flare', as_cmap=True)
    for index, (key, value) in enumerate(dict_arrays.items()):
        # 2d array has to be flattened in order to call it with the dict indices
        ax.flat[index].set(title=key) #, xlabel='coordinate index', ylabel='coordinate index')
        sns.heatmap(value,
                    ax=ax.flat[index],
                    mask=mask, 
                    cmap=cmap, 
                    robust=True, 
                    square=True, 
                    linewidths=0) 
                    #vmin=0, 
                    #vmax=v_max)

    f.suptitle(header, x=0.1, ha='left', size='xx-large')
    plt.show()


def heatmap_single(array, mask_shape, title):
    """
    This function plots a heatmap.
    
    The heatmaps are masked at so that only the lower triangle is drawn.
    
    Arguments
    ---------
    array : numpy object
        data array
    mask_shape : 
        shape of the mask to be applied to the heatmap
    """

    # Generate a mask for the upper triangle from value shape, k=1 to see diagonal line in heatmap
    mask = np.triu(np.ones(mask_shape, dtype=bool), k=0)
    cmap = sns.color_palette("flare", as_cmap=True) # Generate a custom diverging colormap
    #fig, ax = plt.subplot(figsize=(11,9))
    sns.heatmap(array,
                mask=mask, 
                cmap=cmap, 
                robust=True, 
                square=True, 
                linewidths=0,)
    plt.xticks(np.arange(1, array.shape[0]))
    plt.yticks(np.arange(1, array.shape[1]))
    plt.title(title, loc='left')
    plt.show()

    return None


def coord(dict_arrays, xmin, xmax, ymin, ymax):
    """This function plots a set of scatterplots from data stored in a dict."""
    colors = [sns.cubehelix_palette(as_cmap=True),
              sns.cubehelix_palette(start=1.5, rot=-.5, as_cmap=True),
              sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)]

    n_plot = len(dict_arrays)
    f, ax = plt.subplots(1, n_plot, figsize=(18,15))
    sns.set_style("darkgrid")
    aspect = ymax / xmax

    for index, (key, value) in enumerate(dict_arrays.items()):
        ax.flat[index].set(title=key, xlabel='x', ylabel='y', xlim=(xmin,xmax), ylim=(ymin,ymax), 
                           box_aspect=aspect, yticklabels=[], xticklabels=[])
        #ax.flat[index].xaxis.label.set_visible(False)
        #ax.flat[index].yaxis.label.set_visible(False)
        value = np.stack(value, axis=1)
        sns.scatterplot(ax = ax.flat[index], x="0", y="1", 
                        data=pd.DataFrame(value).rename(columns=lambda x: str(x)))
    plt.show()

    return None
