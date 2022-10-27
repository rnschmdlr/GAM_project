"""
Title: Series
Authors: René Schmiedler
Date: 20.08.2021

Description 
-----------
This file allows the user to perfom a number of statistical evaluations and plotting.

Requires
--------
* numpy
* pandas
* seaborn
* matplotlib
* scipy

Functions
---------
* get_means                 - calculate mean for every state.
* plot_distplot             - plot kde plot of joint entropy distribution
* cohen_d                   - calculate effect size measure cohen's d.
* mean_confidence_interval  - calculate 95% confidence interval of an array.
* ttest                     - calculate ttest statistic.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import sqrt
from scipy import stats


def get_means(je_bin_s):
    """
    Calculate mean for every state.

    Arguments
    ---------
    je_bin_s: arr
        sampled joint entropies for a certain bin and state combination

    Return
    ------
    mean: list
    """

    means = [np.mean(arr) for arr in je_bin_s]

    return means


def plot_distplot(means, plot):
    """
    Plot kde plot of joint entropy distribution

    Arguments
    ---------
    means: numpy array 
        means for sampled joint entropy for every bin and state, respectively
    plot: bool 
        flag for whether plot is desired or not

    Return
    ------
    /
    """

    sns.set_style("whitegrid", {'axes.grid' : False})
    colors1 = ["#311B39", "#6486AA", "#265451", "#7BB78C"] # greens and blues
    colors2 = ["#632A6A", "#B54086", "#CE3B56", "#E77C63"] # oranges and purple
    colors12 = ["#311B39", "#6486AA", "#CE3B56", "#E77C63"] # oranges and blues
    colors21 = ["#265451", "#7BB78C", "#632A6A", "#B54086"] # greens and purple
    sns.set_palette(sns.color_palette(colors12))

    df_means = pd.DataFrame(data=means)
    df_means = df_means.reset_index()
    df_means = df_means.rename(columns={0: "locus   4, state 1", 1: "locus 10, state 1",
                                        2: "locus   4, state 3", 3: "locus 10, state 3"})
    df_means = df_means.melt('index', var_name='Label', value_name='Joint entropy')
    #df_means = df_means.drop(['index'], axis=1)
    sns.displot(df_means, x="Joint entropy", hue="Label", kind="kde", fill=False)
    if plot:
        plt.show()


def cohen_d(x, y):
    """
    Calculate effect size measure cohen's d.

    Arguments
    ---------
    x: arr
        first distribution
    y: arr
        second distribution

    Return
    ------
    effect size: float
    """

    return (np.mean(x) - np.mean(y)) / sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2.0)


def mean_confidence_interval(data):
    """
    Calculate 95% confidence interval of an array.

    Arguments
    ---------
    data: arr
        distribution to calculate confidence interval with

    Return
    ------
    check doc
    """

    return stats.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=stats.sem(data))

def ttest(x, y):
    """
    Calculate ttest statistic.

    Arguments
    ---------
    x: arr
        first distribution
    y: arr
        second distribution

    Return
    ------
    check doc
    """

    return stats.ttest(x,y)