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
import numpy as np


def bin_probs(seg_mat, n_bin, hist_len=1):
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
    n_bin : int
        user defined number of bins for each realization
    hist_len : int
        maximum history length from which entropy is measured to be transferred between time series

    Return
    ------
    tprob_all : numpy array
        the probabilities of all bins to transition to other bins in any step of the sequence
        as calculated by calc_probs
    """

    n_tsteps = int(seg_mat.shape[0])
    n_loci = int(seg_mat.shape[2])
    if n_loci % n_bin != 0:
        print('Data not binnable with number of bins chosen.')
        sys.exit()
    else: bin_len = int(n_loci / n_bin)

    # max disrance of any two bins can not be larger than the largest number of occurences of a loci in all slices * number of bins
    global max_dist; max_dist = np.max(np.sum(np.abs(seg_mat), axis=1)) * bin_len * 2
    global min_dist; min_dist = 0
    #global max_dist; max_dist = np.max(seg_mat) + np.abs(np.min(seg_mat))
    #global min_dist; min_dist = 0
    bins_a = np.zeros((n_bin, seg_mat.shape[1], bin_len))
    bins_b = np.zeros((n_bin, seg_mat.shape[1], bin_len))
    trans_probs = np.zeros(shape=(n_tsteps, n_tsteps, n_bin, n_bin))

    # compute all possible state combinations, k off the diagonal
    # remove upper and lower triangular repr. long time intervals with k+1 history length as diagonal cutoff
    mat = np.ones(shape=(n_tsteps, n_tsteps))
    mat_sparse = mat - np.triu(mat, +1 +hist_len) - np.tril(mat, -1)
    comb_vect_sparse = np.transpose(np.nonzero(mat_sparse)) 

    for t_ij in range(len(comb_vect_sparse)):
        tstep_i, tstep_j = comb_vect_sparse[t_ij]

        for idx in range(0, n_loci-bin_len+1, bin_len):
            i = int(idx / bin_len)
            bins_a[i] = seg_mat[tstep_i, :, idx:idx+bin_len]
            bins_b[i] = seg_mat[tstep_j, :, idx:idx+bin_len]

        # TODO make this multi-threaded
        trans_probs[tstep_i, tstep_j] = calc_probs(bins_a, bins_b)
    
    transitions = np.moveaxis(trans_probs, (2,3), (0,1))

    return transitions


def calc_probs(bins_a, bins_b, prnt=False):
    """Computes all element-wise differences of all pairs and the corresponding transition 
    probabilities using a degree matrix."""
    # element-wise differences of all pairs
    dist = bins_a[:, None] - bins_b[None, :]
    dist_mat = np.sum(np.sum(np.abs(dist), axis=-1), axis=-1) 

    # scale matrix from distance 0 to inf -> similarity 1 to 0 so that an all zero distance matrix becomes all ones
    dist_mins = np.subtract(dist_mat, min_dist, out=np.zeros_like(dist_mat), where=dist_mat!=0)
    sim_mat = 1 - ((dist_mins) / (max_dist - min_dist))

    # when there is no difference between two states, the distances must still sum > 0 so that
    # all degrees are > 0 and the degree matrix can be inverted. The transition probability will
    # become distributed equally
    zero_rows = np.where(np.all(np.isclose(sim_mat, 0), axis=1))
    sim_mat[zero_rows] = 1

    outdeg = np.sum(sim_mat, axis=1)[:, None] # weighted degrees as col vector
    deg_mat = np.zeros_like(sim_mat) 
    np.fill_diagonal(deg_mat, outdeg) # degree matrix D

    # T = D^-1 * A
    tprob_bins = np.matmul(np.linalg.inv(deg_mat), sim_mat)

    if prnt:
        print(max_dist)
        #print('\n bins_a: \n', bins_a, '\n bins_b: \n', bins_b)
        print('\n distances: \n', dist_mat)
        print('\n sim mat: \n', sim_mat)
        #print('\n degrees: \n', outdeg)
        print('\n degree matrix: \n', deg_mat)
        print('\n tprob_bins: \n', tprob_bins)

    return tprob_bins
