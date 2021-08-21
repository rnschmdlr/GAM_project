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
* fast_raster - plots a scare numpy array as a raster using datashader
* slow_raster - plots a scare numpy array as a raster using seaborn (for speed comparison)
"""

import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
from datashader import reductions as rd
import xarray as xr
from holoviews.plotting.util import process_cmap
import seaborn as sns


def fast_raster(data):
    """This function plots a scare numpy array as a raster using datashader."""
    df = xr.DataArray(data)
    plot = ds.Canvas(plot_width=1000, plot_height=1000, x_range=(350,500), y_range=(350,500))
    agg = plot.raster(df, interpolate='nearest')

    tf.shade(agg, cmap=process_cmap("Magma", provider="bokeh"), how='eq_hist')

    return None


def slow_raster(data):
    """This function plots a scare numpy array as a raster using seaborn."""
    sns.heatmap(data=pd.DataFrame(data[333:500, 333:500]).rename(columns=lambda x: str(x)),
            cmap=sns.color_palette("flare", as_cmap=True), 
            robust=True, 
            square=True, 
            linewidths=0, 
            cbar_kws={"shrink": .5})
    
    return None
