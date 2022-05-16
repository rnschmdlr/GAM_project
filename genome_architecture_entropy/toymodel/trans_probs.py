"""
Title: Transition probabilities
Authors: René Schmiedler
Date: 20.08.2021

Description 
-----------
This file allows the user to calculate transition probabilities of bins of data points stored 
in a collection of segregation matrices that represent a number of realizations of a 
particular sequence. It can be imported as a module.

Requirements
------------
* numpy

Functions
---------
* bin_probs  - slices entries of a stack of segregation matrices into a number of col bins 
                that are then given to func alc_probs according to pairwise combinations of 
                sequence elemements where transition probabilities are calculated and returned.
* calc_probs - computes all element-wise differences of all pairs and the corresponding transition 
                probabilities using a degree matrix.
"""

import sys
import math

import numpy as np


# %% def functions
def bin_probs(seg_mat, sequence, nbin, hist_len=1):
    """
    This function slices entries of a stack of segregation matrices into a number of col bins 
    that are then given to func calc_probs according to pairwise combinations of sequence 
    elemements where transition probabilities are calculated and returned.
    
    The number of columns of a segregation matrix (i.e number of data points) must be evenly
    devidable by the number of bins desired, otherwise the script is terminated.
    The comb_vect stores all possible combinations of sequence elements such that none appear
    multiple times so that the computational cost is reduced.

    Arguments
    ---------
    seg_mat : numpy array
        collection of segregation matrices (realizations; order of series of no import)
    sequence : numpy vector
        the sequence in which the realizations stored in seg_mat are visited
    nbin : int
        user defined number of bins for each realization
    hist_len : int
        maximum history length from which entropy is measured to be transferred between time series

    Return
    ------
    tprob_all : numpy array
        the probabilities of all bins to transition to other bins in any step of the sequence
        as calculated by calc_probs
    """

    seq_len = int(seg_mat.shape[2])
    if seq_len % nbin != 0:
        print('Data not binnable with number of bins chosen.')
        sys.exit()
    else: bin_len = int(seq_len / nbin)

    bins1 = np.zeros((nbin, seg_mat.shape[1], bin_len))
    bins2 = np.zeros((nbin, seg_mat.shape[1], bin_len))
    tprob_all = np.empty(shape=(0, nbin))

    # compute all possible state combinations
    comb_vect = np.array(np.meshgrid(sequence, sequence)).T.reshape(-1,2)
    comb_vect = np.unique(comb_vect, axis=0)
    n = int(math.sqrt(comb_vect.shape[0]))

    # only compute state combinations of k off the diagonal
    l = len(sequence)
    mat = np.ones((l, l))
    # remove upper and lower triangular repr. long time intervals with k+1 history length as diagonal cutoff
    mat_sparse = mat - np.triu(mat, hist_len +1) - np.tril(mat, -hist_len -1)
    comb_vect_sparse = np.transpose(np.nonzero(mat_sparse)) 

    # checks which elements from comb_vect_sparse are in comb_vect and returns bool vector
    check = (comb_vect[:, None] == comb_vect_sparse).all(-1).any(-1)

    for zx in range(comb_vect.shape[0]):
        # only compute transition probability for non zero combinations -> eval true in check
        if check[zx]:
            state1 = int(comb_vect[zx, 0])
            state2 = int(comb_vect[zx, 1])

            for idx in range(0, seq_len-bin_len+1, bin_len):
                i = int(idx / bin_len)
                bins1[i] = seg_mat[state1, :, idx:idx+bin_len]
                bins2[i] = seg_mat[state2, :, idx:idx+bin_len]

            #print('\n__________________________\n', state1, state2)
            tprob_bin = calc_probs(bins1, bins2)
            
        else:
            tprob_bin = np.zeros(shape=(nbin, nbin))
        
        tprob_all = np.append(tprob_all, tprob_bin, axis=0)

    tprob_all = np.reshape(tprob_all, (n, n, nbin, nbin))

    return tprob_all


def calc_probs(bins1, bins2, prnt=False):
    """Computes all element-wise differences of all pairs and the corresponding transition 
    probabilities using a degree matrix."""
    # element-wise differences of all pairs
    dist = bins1[:, None] - bins2[None, :] 
    dist_mat = np.sum(np.sum(np.abs(dist), axis=-1), axis=-1) 
    
    # scale matrix unless it is all 0. In that case it becomes all ones.
    if dist_mat.max() != 0:
        dist_mat = dist_mat - dist_mat.min()
        dist_mat = 1 - dist_mat/dist_mat.max()
    else: dist_mat = np.ones_like(dist_mat)

    # when there is no difference between two states, the distances must still sum > 0 so that
    # all degrees are > 0 and the degree matrix can be inverted. The transition probability will
    # become distributed equally
    zero_rows = np.where(np.all(np.isclose(dist_mat, 0), axis=1))
    dist_mat[zero_rows] = 1

    outdeg = np.sum(dist_mat, axis=1)[:, None] # weighted degrees as col vector
    deg_mat = np.zeros_like(dist_mat) 
    np.fill_diagonal(deg_mat, outdeg) # degree matrix D

    # M = D^-1 * A
    tprob_bins = np.matmul(np.linalg.inv(deg_mat), dist_mat)

    if prnt:
        print('\n bins1: \n', bins1, '\n bins2: \n', bins2)
        print('\n distances: \n', dist)
        print('\n dist mat: \n', dist_mat)
        print('\n degrees: \n', outdeg)
        print('\n degree matrix: \n', deg_mat)
        print('\n tprob_bins: \n', tprob_bins)

    return tprob_bins
