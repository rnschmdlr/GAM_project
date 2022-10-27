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


def plot_ensemble(coords_model, je_mat, mi_mat, coseg_mat, title, entropy, color_points, vmax):
    '''2x2 plot fxn'''
    # Generate a mask for the upper triangle from value shape, k=1 to see diagonal line in heatmap
    mask_shape = (je_mat.shape[1], je_mat.shape[1])
    mask = np.triu(np.ones(mask_shape, dtype=bool), k=0)

    # style and colors
    sns.set_theme(style='white')
    cmap = sns.color_palette('flare', as_cmap=True)
    scatter_cmap = sns.color_palette('dark:dodgerblue_r', as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=sns.color_palette('dark:dodgerblue', as_cmap=True))

    # dynamic coordinate boundaries
    xmin = np.min(coords_model[0])
    xmax = np.max(coords_model[0])
    ymin = np.min(coords_model[1])
    ymax = np.max(coords_model[1])
    height = ymax - ymin
    width = xmax - xmin
    
    if max(height, width) == width:
        ymax = 1/2 * width
        ymin = -1/2 * width
        f = 0.08 * width # spacing factor
    else:
        xmax = 1/2 * height
        xmin = -1/2 * height
        f = 0.08 * height

    dyn_boundaries = [xmin - f, xmax + f, ymin - f, ymax + f]

    #static coordinate boundaries
    xnum = coords_model.shape[1]
    f = 0.08 * xnum
    ymin = -1/2 * xnum
    ymax = 1/2 * xnum
    boundaries = [0 - f, xnum + f, ymin - f, ymax + f]
    #boundaries = np.mean( np.array([ dyn_boundaries, static_boundaries ]), axis=0 )

    # set figure
    fig, (ax1, ax2) = plt.subplots(2, 2)
    fig.set_size_inches(12, 10)
    fig.suptitle(title, fontsize=25, weight='bold')
    
    # set coordinate plot
    ax1[0].plot(coords_model[0], coords_model[1], '-k') 
    ax1[0].scatter(coords_model[0], coords_model[1], s=50, c=color_points, cmap=scatter_cmap)
    ax1[0].axis(boundaries)
    ax1[0].tick_params(labelleft=False, labelbottom=False)
    ax1[0].set_title('Chromatin model colored by accumulated MI', loc='left', fontsize=16)
    ax1[0].axis('off')
    plt.colorbar(sm, ax=ax1[0], shrink=0.5).outline.set_visible(False)

    # set 3 heatmap plots
    ax1[1].set_title('Mutual Information (MI)', loc='left', fontsize=16)
    ax1[1].tick_params(labelleft=False, labelbottom=False)
    sns.heatmap(mi_mat,
                ax=ax1[1],
                mask=mask,
                cmap=cmap,
                robust=True,
                square=True,
                linewidths=0,
                cbar_kws={"shrink": .5},
                vmax=vmax[0],
                vmin=0)
    
    ax2[0].set_title('Linkage disequilibirum (LD)', loc='left', fontsize=16)
    ax2[0].tick_params(labelleft=False, labelbottom=False)
    sns.heatmap(coseg_mat,
                ax=ax2[0],
                mask=mask, 
                cmap=cmap, 
                robust=True, 
                square=True, 
                linewidths=0,
                cbar_kws={"shrink": .5},
                vmax=vmax[1])
    
    ax2[1].set_title('Normalized difference (LD - MI)', loc='left', fontsize=16)
    ax2[1].tick_params(labelleft=False, labelbottom=False)
    sns.heatmap(je_mat,
                ax=ax2[1],
                mask=mask, 
                cmap=cmap, 
                robust=True, 
                square=True, 
                linewidths=0,
                cbar_kws={"shrink": .5},
                vmax=vmax[2],
                vmin=0)

    plt.figtext(0.4, 0.05, 'Global entropy H = '+entropy, fontsize=14)
    plt.show()