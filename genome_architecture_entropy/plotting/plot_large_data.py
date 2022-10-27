"""
Title: Plot Large Data
Authors: René Schmiedler
Date: 20.08.2021

Description 
-----------
This file allows the user to plot large numpy array efficiently using datashader and can be 
imported as a module.

Requires
--------
* pandas
* datashader
* xarray
* holoviews
* seaborn

Functions
---------
* fast_raster - plots a numpy array as a raster using datashader
* slow_raster - plots a numpy array as a raster using seaborn (for speed comparison)
"""

# %%
import matplotlib
import pandas as pd
import numpy as np
import datashader as ds
import datashader.transfer_functions as tf
from datashader import reductions as rd
import xarray as xr
from holoviews.plotting.util import process_cmap
import seaborn as sns
import matplotlib.pyplot as plt

import cmcrameri.cm as cmc
import colorcet as cc
import cmasher as cmr


def fast_raster(data):
    """This function plots a numpy array as a raster using datashader."""
    global df
    df = xr.DataArray(data)
    plot = ds.Canvas(plot_width=1000, plot_height=1000, x_range=(350,500), y_range=(350,500))
    agg = plot.raster(df, interpolate='nearest')

    tf.shade(agg, cmap=process_cmap("Magma", provider="bokeh"), how='eq_hist')
    
    return None


def slow_raster(data, vmin, vmax, xticklabels, yticklabels, title):
    """This function plots a numpy array as a raster using seaborn."""
    #sns.heatmap(data=pd.DataFrame(data[333:500, 333:500]).rename(columns=lambda x: str(x)),
    plt.rcParams["figure.figsize"] = (7,7)
    ax = sns.heatmap(data=data,
            cmap=cmc.oslo_r,
            #cmap=sns.color_palette("blend:ghostwhite,gainsboro,lightsteelblue,steelblue,midnightblue", as_cmap=True), 
            #cmap=matplotlib.cm.get_cmap('bwr'),
            #cmap=cmc.fes,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            robust=True, 
            square=True, 
            linewidths=0, 
            vmin=vmin,
            vmax=vmax,
            center=None,
            cbar_kws={"shrink": .82})
    ax.set_title(title)
    ax.invert_yaxis()

    #plt.savefig('/Users/pita/Documents/Rene/GAM_project/figures/' + title + '.png', dpi=150)
    plt.show()

    return None


def lineplot(windows, methods, *args):
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams['figure.dpi'] = 150
    sns.set_theme(style="whitegrid")

    fig = sns.lineplot(data=windows.melt(id_vars=['visited', "window size"], value_vars=methods, var_name="method"),
                        x="visited", 
                        y="value",
                        hue="method", 
                        style="window size",
                        linewidth=1.5, 
                        #palette="Set2",
                        palette=["#4A87E3", "#E49AC3"]) #"#B695F7"])

    #fig, ax = plt.subplots()
    #ax.plot(data, linewidth=2.5)
    #ax.fill_between(x=windows.index, y1=data["mje"]-std["je_std"], y2=data["mje"]+std["je_std"], alpha=0.2)
    #ax.fill_between(x=windows.index, y1=data["mmi"]-std["mi_std"], y2=data["mmi"]+std["mi_std"], alpha=0.2)
    #ax.fill_between(x=windows.index, y1=data["mncmi"]-std["ncmi_std"], y2=data["mncmi"]+std["ncmi_std"], alpha=0.2)
    #ax.fill_between(x=windows.index, y1=data["mnpmi"]-std["npmi_std"], y2=data["mnpmi"]+std["npmi_std"], alpha=0.2)

    #plt.vlines([195, 377, 537, 694, 846, 996, 1141, 1271, 1395, 1526, 1648, 1768, 1889, 2014, 2118, 2216, 2311, 2401, 2463, 2632], 0.1, 0.6)
    plt.xlabel("Chromatin region [Mbp]")
    plt.ylabel("Mean MI [bit]")
    plt.ylim(0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.savefig('/Users/pita/Documents/Rene/GAM_project/figures/' + args[0], dpi=150)
    plt.show(fig)
    

    return None
#data = np.load('/Users/pita/Documents/Rene/GAM_project/genome_architecture_entropy/data/segregation_at_30kb.table_MI.npy')
#slow_raster(data)