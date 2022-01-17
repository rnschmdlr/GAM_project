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
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import stats

import entropy_measures as em
import toymodel.conf_space as cs
import toymodel.series as scs
import toymodel.trans_probs as tb
import plotting.plot_large_data as pld
import plotting.plot_small_data as psd
import toymodel.stat_evaluation as se


def load_data(filename):
    """Load segregation table from file and convert to numpy array"""
    col_to_skip = ['chrom', 'start', 'stop']
    segtab = pd.read_table(filename, usecols=lambda x: x not in col_to_skip)
    seg_mat = np.array(segtab, dtype=bool)

    return seg_mat


def toy_models_simple(samples=1, plot=True, stats=False):
    """Defines a set of states as coordinates on an xy plane and saves them to 
    a dict for convenient plotting and a stacked array for further operations"""
    closed_state = cs.rotate(np.array([[5.0, 5.0, 5.0, 5.2, 6.1, 7.0, 7.5, 7.0, 6.1, 5.2, 5.0, 5.0, 5.0],
                                       [1.5, 2.5, 3.5, 4.3, 4.0, 4.1, 5.0, 5.8, 6.0, 5.7, 6.5, 7.5, 8.5]]), 270)

    open_state = cs.rotate(np.array([[5.0, 5.0, 5.0, 5.2, 6.1, 7.0, 7.5, 7.0, 6.1, 5.2, 5.0, 5.0, 5.0],
                                     [0.0, 1.0, 2.0, 2.9, 3.2, 3.8, 5.0, 6.2, 6.8, 7.1, 8.0, 9.0, 10.0]]), 270)
    intermediate_state = np.mean(np.array([closed_state, open_state]), axis=0)

    conf_arr = np.stack([closed_state, intermediate_state, open_state], axis=0)

    if stats:
        s1 = cs.ran_conf_space(samples, closed_state)
        s2 = cs.ran_conf_space(samples, open_state)

        return s1, s2
    
    if plot:
        conf_dict = {'State 1': closed_state,
                     'State 2': intermediate_state,
                     'State 3': open_state}
        psd.coord(conf_dict, -5, 15, -10, 10)

    return conf_arr


def toy_model_complex(size, plot):
    """
    Defines a model with a beginning and end state and computes in between states
    
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

    chain1 = [0,0, 1,0, 2,0, 3,0, 4,0, 5,0, 6,0, 7,0, 8,0, 9,0, 10,0, 11,0, 12,0, 13,0, 14,0, 15,0, 
              16,0, 17,0, 18,0, 19,0, 20,0, 21,0, 22,0, 23,0, 24,0, 25,0, 26,0, 27,0, 28,0, 29,0, 
              30,0, 31,0, 32,0, 33,0, 34,0, 35,0]

    chain2 = [116,110,139,99,129,77,134,55,158,55,162,77,150,98,173,105,161,126,152,148,157,171,180,
              175,191,153,192,130,187,107,208,95,226,108,222,131,211,151,214,175,227,194,252,193,265,
              174,262,151,251,130,241,109,262,98,247,80,250,57,266,41,283,56,285,79,275,98,293,111,
              316,112,340,108]

    rawcoords = np.array([chain1, chain2])
    n, k = rawcoords.shape
    states = np.empty((n,2, int(k/2)))
    l1, l2 = 38, 138 # real world lengths of the chains

    for s in range(n):
        # devide sequence into two array for x, y
        states[s] = np.array([rawcoords[s,0::2], rawcoords[s,1::2]]) 
        #states[s] = rotate(states[s], -45, origin=states[s].mean(axis=1, keepdims=True)) 
        if s==1:
            states[s] = states[s] - states[s].min(axis=1, keepdims=True) # +1
            states[s] = states[s] / states[s].max(axis=1, keepdims=True) * states[s-1].max()
            # scale chain in order to preserve loci distance
            states[s] = states[s] * l1/l2 
            # center horizontally relative to chain 1
            states[s,0] = states[s,0] + states[s-1,0].max()/2 - states[s,0].mean() 
            states[s,1] = states[s,1] - states[s,1].mean()

    if plot:
        #psd.plot_xy(states[s], 0, 35, -10, 10)
        psd.coord({'State 0': states[0], 'State end': states[-1]}, 0, 35, -10, 10)

    seg_mat = (cs.slice(states[1], 0, 35, -10, 10))
    je_mat1 = em.all_joint_entropy(seg_mat.T)
    mi_mat1 = em.mutual_information(je_mat1)
    dict_entropy = {'MI state end': mi_mat1}
    #psd.heatmap(dict_entropy, 'JE vs MI', v_max=0)

    seg_mats = scs.series_conf_space(size, states[0], states[1])
    
    return seg_mats


def pipe_je_mi_toy(simple_model, plot=True, prnt_stats=True):
    """An encapsulation to apply func next_pipe to a collection of states and stack results for plotting"""

    def print_stats(array_dict):
        """A helper function to print mean and standart deviation of matrices."""
        np.set_printoptions(suppress=True, precision=2, sign=' ', linewidth=100)
        pd.set_option('display.float_format', lambda x: '%.2f' % x)

        for name, val in array_dict.items(): #known issue: res no longer returned as dict
            print(name, val, sep='\n')
            print(pd.DataFrame(val).describe().loc[['mean', 'std']])
        
        return None
    

    def next_pipe(xy_model):
        """
        A pipeline to compute the joint entropy and mutual information.
        
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
        seg_mat = cs.slice(xy_model, 0, 10, 0, 10)
        je_mat1 = em.all_joint_entropy(seg_mat.T)
        mi_mat1 = em.mutual_information(je_mat1)

        # Compute entropy measures from distance based input (pw point distance matrix)
        #dist    = distance.cdist(np.stack(xy_model, axis=1), np.stack(xy_model, axis=1))
        #je_mat2 = em.all_joint_entropy(dist)
        #mi_mat2 = em.mutual_information(je_mat2)

        #results = {'\nJE of windows': je_mat1, 
        #           '\nJE of distances': je_mat2, 
        #           '\nMI of windows': mi_mat1, 
        #           '\nMI of distances': mi_mat2}

        v_max = np.max(np.concatenate([je_mat1, mi_mat1]))
        results = np.stack([je_mat1, mi_mat1], axis=0)
                
        return results, v_max

    # Joint entropy and mutual information applied to three-state toy data
    je_mi = np.empty(shape=(0, 36, 36)) #13 for og simple model

    # stack results
    for states in range(0, simple_model.shape[0]):
        res, v_max = next_pipe(simple_model[states])
        je_mi = np.concatenate([je_mi, res])

        if prnt_stats:
            print_stats(res)

    # store to dict to allow for subtitles
    state = ['JE state 1', 'MI state 1', 'JE state 2', 'MI state 2', 'JE state 3', 'MI state 3'] #np.arange(je_mi.shape[0]) + 1
    dict_entropy = dict(zip(state, je_mi))

    if plot:
        psd.heatmap(dict_entropy, 'JE vs MI', v_max)
        
        
def pipe_trans_entropy(model, series_length, nbin, plot=True, prnt=True):
    """Transfer entropy applied to generated time series data"""
    start = timeit.default_timer()
    
    sequence = scs.generate_series(model, series_length, prnt)
    probs = tb.bin_probs(model, sequence, nbin=nbin)
    trans_entropy = em.all_transfer_entropy(probs) 

    if plot:
        g = sns.heatmap(data=pd.DataFrame(trans_entropy).rename(columns=lambda x: str(x)),
                cmap=sns.color_palette("flare", as_cmap=True), 
                robust=True, 
                square=True, 
                linewidths=0)
        g.set(xticklabels=np.arange(1,37,2))#trans_entropy.shape[0]+1))
        g.set(yticklabels=np.arange(1,36,3))#trans_entropy.shape[0]+1))
        plt.show(block=False)
        

    if prnt:
        print('\n Trans-Entropy of bins: \n', trans_entropy)  
        print('\n Runtime: ', timeit.default_timer() - start) 


def pipe_je_mi_exp_data(file, plot=True, save=True):
    """Joint entropy and mutual information applied to experimental GAM data"""
    # TODO implement range
    seg_mat = load_data('data/'+file)

    je_mat = em.all_joint_entropy(seg_mat)
    mi_mat = em.mutual_information(je_mat)
    
    if save:
        np.save(file+'_JE.npy', je_mat)
        np.save(file+'_MI.npy', mi_mat)

    if plot:
        pld.fast_raster(je_mat)
        pld.fast_raster(mi_mat)

    return je_mat, mi_mat


def call_je_stats(samples, loc1, loc2):
    """Helper function that calls statistical tests for 2 loci from a number of samples of the simple model."""

    def je_stats_for_two_loci(s1, s2, bins, samples, bin1, bin2):
        """This function performs entropy calcualtions for the test cases."""
        je3_s1 = je3_s2 = je9_s1 = je9_s2 = np.empty(0)

        for states in range(0, s1.shape[0]):
            je_mat1 = em.all_joint_entropy(s1[states])
            je_mat2 = em.all_joint_entropy(s2[states])

            je3_s1 = np.concatenate((je3_s1, je_mat1[bin1, :]), axis=0)
            je3_s2 = np.concatenate((je3_s2, je_mat2[bin1, :]), axis=0)

            je9_s1 = np.concatenate((je9_s1, je_mat1[bin2, :]), axis=0)
            je9_s2 = np.concatenate((je9_s2, je_mat2[bin2, :]), axis=0)

        je3_s1 = je3_s1.reshape(samples, bins)
        je3_s2 = je3_s2.reshape(samples, bins)

        je9_s1 = je9_s1.reshape(samples, bins)
        je9_s2 = je9_s2.reshape(samples, bins)

        return je3_s1, je9_s1, je3_s2, je9_s2


    s1, s2 = toy_models_simple(samples, plot=False, stats=True)

    print('Calculating joint entropy...')
    je_bin1_s1, je_bin2_s1, je_bin1_s2, je_bin2_s2 = je_stats_for_two_loci(s1, s2, bins=s1.shape[2],
                                                                           samples=samples,
                                                                           bin1=loc1,
                                                                           bin2=loc2)

    print('Calculatig means...')
    all_means = [se.get_means(je) for je in zip(je_bin1_s1, je_bin2_s1, je_bin1_s2, je_bin2_s2)]

    # histogram of joint entropy distribution
    se.plot_distplot(np.array(all_means), plot=True)

    print('ttest')
    print(stats.ttest_rel(je_bin1_s1, je_bin1_s2))
    print(stats.ttest_rel(je_bin2_s1, je_bin2_s2))

    print('effect size')
    print(se.cohen_d(je_bin1_s1, je_bin1_s2))
    print(se.cohen_d(je_bin2_s1, je_bin2_s2))


if __name__ == '__main__': 
    np.set_printoptions(suppress=True, precision=2, sign=' ', linewidth=150, threshold=500)

    # applies joint entropy and mutual information to window sampling of a simple 
    # three state TAD model
    print('setting simple model...')
    simple_model = toy_models_simple(plot=True, stats=False)
    print('calculating entropies...')
    pipe_je_mi_toy(simple_model, plot=True, prnt_stats=False)

    # print('Statistical evaluation of entropy measures.')
    # call_je_stats(samples=5, loc1=3, loc2=9)

    # applies transfer entropy to a more complex TAD model: a series of varying length
    print('setting complex model...')
    complex_model = toy_model_complex(size=20, plot=True)
    print('calculating trans entropy...')
    pipe_trans_entropy(complex_model, series_length=20, nbin=36, plot=True, prnt=True)
        
    # applies joint entropy and mutual information to experimental GAM data
    file = 'segregation_at_30kb.table'
    print('calculating entropies of ', file)
    pipe_je_mi_exp_data(file, plot=True, save=False)
# %%

