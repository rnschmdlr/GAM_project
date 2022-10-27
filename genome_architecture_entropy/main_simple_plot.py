"""
Title: Genome Architecture Entropy
Authors: René Schmiedler
Date: 20.08.2021

Description 
-----------
This script allows the user to apply information theoretic measures to 
either pre-existing genomic segregation data or to generate a 
simplified model of TAD formation from coordinates that are already 
supplied.

Requirements
------------
see requirements.txt
"""

# %% main script
import numpy as np
import pandas as pd
import seaborn as sns
import timeit

from matplotlib import pyplot as plt
from scipy.spatial import distance

import entropy_measures as em
import toymodel.conf_space as cs
import toymodel.series as scs
import toymodel.trans_probs as tb
import plotting.plot_large_data as pld
import plotting.plot_small_data as psd


def load_data(filename):
    """Load segregation table from file and convert to numpy array"""
    col_to_skip = ['chrom', 'start', 'stop']
    segtab = pd.read_table(filename, usecols=lambda x: x not in col_to_skip)
    seg_mat = np.array(segtab, dtype=bool)

    return seg_mat


def toy_models_simple(plot):
    """Defines a set of states as coordinates on an xy plane and saves them to 
    a dict for convenient plotting and a stacked array for further operations"""
    closed_state = cs.rotate(np.array([[5.0, 5.0, 5.0, 5.2, 6.1, 7.0, 7.5, 7.0, 6.1, 5.2, 5.0, 5.0, 5.0],
                                       [1.5, 2.5, 3.5, 4.3, 4.0, 4.1, 5.0, 5.8, 6.0, 5.7, 6.5, 7.5, 8.5]]), 270)

    open_state = cs.rotate(np.array([[5.0, 5.0, 5.0, 5.2, 6.1, 7.0, 7.5, 7.0, 6.1, 5.2, 5.0, 5.0, 5.0],
                                     [0.0, 1.0, 2.0, 2.9, 3.2, 3.8, 5.0, 6.2, 6.8, 7.1, 8.0, 9.0, 10.0]]), 270)
    intermediate_state = np.mean(np.array([closed_state, open_state]), axis=0)

    conf_dict = {'State 1': closed_state,
                 'State 2': intermediate_state,
                 'State 3': open_state}

    conf_arr = np.vstack([cs.ran_conf_space(1, closed_state),
                          cs.ran_conf_space(1, intermediate_state),
                          cs.ran_conf_space(1, open_state)])

    conf_arr = np.stack([closed_state, intermediate_state, open_state], axis=0)

    if plot:
        psd.coord(conf_dict, 0, 10, 0, 10)

    return conf_arr


def toy_model_complex(size, plot):
    """Defines a model with a beginning and end state and computes in between states
    
    Raw xy-coordinates are first normalized then aligned. The in between states are 
    computed using a weighted mean method according to the size of the state or 
    configuration space. The output is already sliced (see fun slice in conf_space.py).

    Arguments
    ---------
    size : int
        number of states to be generated
        
    plot : bool
        whether or not to plot the start and end states
        
    Return
    ------
    seg_mats : numpy array of shape (size, 1000, 36)
        stacked segregation matrices of the sliced configurations in the model
    """

    chain1 = [0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0,
              16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0, 25, 0, 26, 0, 27, 0, 28, 0, 29, 0,
              30, 0, 31, 0, 32, 0, 33, 0, 34, 0, 35, 0]

    chain2 = [116, 110, 139, 99, 129, 77, 134, 55, 158, 55, 162, 77, 150, 98, 173, 105, 161, 126, 152, 148, 157, 171,
              180,
              175, 191, 153, 192, 130, 187, 107, 208, 95, 226, 108, 222, 131, 211, 151, 214, 175, 227, 194, 252, 193,
              265,
              174, 262, 151, 251, 130, 241, 109, 262, 98, 247, 80, 250, 57, 266, 41, 283, 56, 285, 79, 275, 98, 293,
              111,
              316, 112, 340, 108]

    rawcoords = np.array([chain1, chain2])
    n, k = rawcoords.shape
    states = np.empty((n, 2, int(k / 2)))
    l1, l2 = 38, 138  # real world lengths of the chains

    for s in range(n):
        # devide sequence into two array for x, y
        states[s] = np.array([rawcoords[s, 0::2], rawcoords[s, 1::2]])
        # states[s] = rotate(states[s], -45, origin=states[s].mean(axis=1, keepdims=True))
        if s == 1:
            states[s] = states[s] - states[s].min(axis=1, keepdims=True)  # +1
            states[s] = states[s] / states[s].max(axis=1, keepdims=True) * states[s - 1].max()
            # scale chain in order to preserve loci distance
            states[s] = states[s] * l1 / l2
            # center horizontally relative to chain 1
            states[s, 0] = states[s, 0] + states[s - 1, 0].max() / 2 - states[s, 0].mean()
            states[s, 1] = states[s, 1] - states[s, 1].mean()

    if plot:
        # psd.plot_xy(states[s], 0, 35, -10, 10)
        psd.coord({'State 0': states[0], 'State end': states[-1]}, 0, 35, -10, 10)

    seg_mats = scs.series_conf_space(size, states[0], states[1])

    return seg_mats


def pipe_je_mi_toy(simple_model, plot=True, prnt_stats=True):
    """An encapsulation to apply func next_pipe to a collection of states"""

    def print_stats(array_dict):
        """A helper function to print mean and standart deviation of matrices."""
        np.set_printoptions(suppress=True, precision=2, sign=' ', linewidth=100)
        pd.set_option('display.float_format', lambda x: '%.2f' % x)

        for name, val in array_dict.items():
            print(name, val, sep='\n')
            print(pd.DataFrame(val).describe().loc[['mean', 'std']])

        return None

    def next_pipe(xy_model):
        """A pipeline to compute the joint entropy and mutual information.
        
        Two different distance measures are used: a segregation by slicing
        and eucledian distance of points. The results are stored in a dictionary
        to be unpacked together for comparison by plotting.
        
        Arguments
        ---------
        xy_model : numpy array
            two row vectors storing x and y coordinates respectively
        
        Return
        ------
        results : dict
            four entries: a combination of each entropy and distance measure
        """

        # Slice xy testmodel into segregation windows and compute entropy measures
        seg_mat = (cs.slice(xy_model, 0, 10, 0, 10))
        je_mat1 = em.all_joint_entropy(seg_mat.T)
        mi_mat1 = em.mutual_information(je_mat1)

        # Compute entropy measures from distance based input (pw point distance matrix)
        dist = distance.cdist(np.stack(xy_model, axis=1), np.stack(xy_model, axis=1))
        # je_mat2 = em.all_joint_entropy(dist)
        # mi_mat2 = em.mutual_information(je_mat2)
        # get max for legend labeling
        v_max = np.max(np.concatenate([je_mat1, mi_mat1]))

        results = np.stack([je_mat1, mi_mat1], axis=0)

        return results, v_max

    # Joint entropy and mutual information applied to three-state toy data
    global je_mi
    je_mi = np.empty(shape=(0, 13, 13))
    state = [np.arange(simple_model.shape[0])]

    for states in range(0, simple_model.shape[0]):
        global res
        global v_max
        res, v_max = next_pipe(simple_model[states])
        je_mi = np.concatenate([je_mi, res])
    # num_states = simple_model.shape[0]
    state = np.arange(je_mi.shape[0]) + 1
    global dict_entropy
    dict_entropy = dict(zip(state, je_mi))

    if plot:
        psd.heatmap(dict_entropy, 'State', v_max)

    if prnt_stats:
        print_stats(res)



def pipe_trans_entropy(model, series_length, plot=True, prnt=True):
    """Transfer entropy applied to generated time series data"""
    start = timeit.default_timer()

    sequence = scs.generate_series(model, series_length, prnt)
    probs = tb.bin_probs(model, sequence, nbin=36)
    trans_entropy = em.all_transfer_entropy(probs)

    if plot:
        sns.heatmap(data=pd.DataFrame(trans_entropy).rename(columns=lambda x: str(x)),
                    cmap=sns.color_palette("flare", as_cmap=True),
                    robust=True,
                    square=True,
                    linewidths=0,
                    cbar_kws={"shrink": .5})

    if prnt:
        print('\n Trans-Entropy of bins: \n', trans_entropy)
        print('\n Runtime: ', timeit.default_timer() - start)


def pipe_je_mi_exp_data(file, plot=True, save=True):
    """Joint entropy and mutual information applied to experimental GAM data"""
    # TODO implement range
    seg_mat = load_data('data/' + file)

    je_mat = em.all_joint_entropy(seg_mat)
    mi_mat = em.mutual_information(je_mat)

    if save:
        np.save(file + '_JE.npy', je_mat)
        np.save(file + '_MI.npy', mi_mat)

    if plot:
        pld.fast_raster(je_mat)
        pld.fast_raster(mi_mat)

    return je_mat, mi_mat


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=2, sign=' ', linewidth=150, threshold=500)

    # for testing purposes
    test_seg = np.array([[[0, 1, 1, 1], [0, 1, 0, 0]],
                         [[1, 1, 1, 1], [1, 0, 0, 1]],
                         [[1, 0, 0, 1], [1, 1, 0, 0]],
                         [[1, 1, 0, 1], [1, 1, 1, 0]]])

    # applies joint entropy and mutual information to distance and window sampling of a simple 
    # three state TAD model
    simple_model = toy_models_simple(plot=False)
    pipe_je_mi_toy(simple_model, plot=True, prnt_stats=False)

    # applies transfer entropy to a more complex TAD model: a series of varying length
    # complex_model = toy_model_complex(size=20, plot=True)
    # pipe_trans_entropy(complex_model, series_length=20, plot=True, prnt=True)

    # applies joint entropy and mutual information to experimental GAM data
    # file = 'segregation_at_30kb.table'
    # pipe_je_mi_exp_data(file, plot=True, save=True)
