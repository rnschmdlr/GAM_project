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


def plot_xy(mat, xmin, xmax, ymin, ymax):
    "This is a simple plotting helper function for xy coordinates."
    plt.show(block=False)
    plt.scatter(mat[0], mat[1])
    plt.axis([xmin, xmax, ymin, ymax])
    plt.gca().set_aspect('equal')
    plt.show()


def heatmap(dict_arrays, header, mask_shape):
    """This function plots a set of heatmaps from data stored in a dict.
    
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
    mask = np.triu(np.ones(mask_shape, dtype=bool), k=0)
    f, ax = plt.subplots(2, 2, figsize=(11, 9)) # Set up the matplotlib figure
    cmap = sns.color_palette("flare", as_cmap=True) # Generate a custom diverging colormap

    for index, (key, value) in enumerate(dict_arrays.items()):
        # 2d array has to be flattened in order to call it with the dict indices
        ax.flat[index].set(title=key, xlabel='coordinate index', ylabel='coordinate index')
        sns.heatmap(value, 
                    ax=ax.flat[index],
                    mask=mask, 
                    cmap=cmap, 
                    robust=True, 
                    square=True, 
                    linewidths=0, 
                    cbar_kws={"shrink": .5})

    f.suptitle(header, x=0.1, ha='left', size='xx-large')
    plt.show

    return None


def coord(dict_arrays, xmin, xmax, ymin, ymax):
    """This function plots a set of scatterplots from data stored in a dict."""
    colors = [sns.cubehelix_palette(as_cmap=True),
              sns.cubehelix_palette(start=1.5, rot=-.5, as_cmap=True),
              sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)]

    n_plot = len(dict_arrays)
    f, ax = plt.subplots(1, n_plot, figsize=(11,9))
    sns.set_style("darkgrid")
    aspect = ymax / xmax

    for index, (key, value) in enumerate(dict_arrays.items()):
        ax.flat[index].set(title=key, xlabel=' ', ylabel=' ', xlim=(xmin,xmax), ylim=(ymin,ymax), 
                           box_aspect=aspect, yticklabels=[], xticklabels=[])
        ax.flat[index].xaxis.label.set_visible(False)
        ax.flat[index].yaxis.label.set_visible(False)
        value = np.stack(value, axis=1)
        sns.scatterplot(ax = ax.flat[index], x="0", y="1", 
                        data=pd.DataFrame(value).rename(columns=lambda x: str(x)))
    plt.show()

    return None
