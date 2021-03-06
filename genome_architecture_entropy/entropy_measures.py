"""
Title: Entropy Measures
Authors: René Schmiedler
Date: 20.08.2021

Description 
-----------
This file allows the user to apply information theoretic measures to 
genomic segregation data and can be imported as a module.

Requirements
------------
* numpy

Functions
---------
* shannon_entropy       - returns total entropy of a matrix
* joint_entropy         - returns joint entropy of two variables in a matrix
* all_joint_entropy     - returns joint entropy of all combination of row vectors in a matrix
* mutual_information    - returns mutual information of a matrix given a joint entropy matrix
* transfer_entropy      - returns transfer entropy of two probability distributions 
* all_transfer_entropy  - returns transfer entropy of all combinations of probability distributions
"""

import numpy as np
import warnings
    

def shannon_entropy(seg_mat):
    """Returns the total entropy of a matrix"""
    freq = np.unique(seg_mat, return_counts=True)[1] / np.prod(seg_mat.shape)
    freq = freq[freq != 0] # drop 0 to avoid Nan resulting from log2
    h = -np.sum(freq * np.log2(freq))

    return h


def joint_entropy(seg_mat, w_x, w_y):
    """Returns the joint entropy of two variables in a matrix"""
    join = np.array(np.vstack((seg_mat[w_x,], seg_mat[w_y,]))) # too slow to make new array?

    # count true in intersect
    p00 = np.count_nonzero(np.logical_and(join[0,]==0, np.equal(join[0,], join[1,]))) 
    p11 = np.count_nonzero(np.logical_and(join[0,]==1, np.equal(join[0,], join[1,]))) 

    p01 = np.size(join[0,]) - np.count_nonzero(np.logical_or(join[0,]==1, np.equal(join[0,], join[1,])))
    p10 = np.size(join[0,]) - np.count_nonzero(np.logical_or(join[0,]==0, np.equal(join[0,], join[1,])))

    props = np.array([p00, p01, p10, p11]) / join.shape[1] # devide by length of row
    props = props[props != 0]

    h = -np.sum(props * np.log2(props))

    return h


def all_joint_entropy(seg_mat):
    """Returns the joint entropy of all combination of row vectors in a matrix"""
    # TODO vectorize this to increase speed 
    # idea1 use np.triu_indices to create a mask for an upper triangle
    # idea2 numba compatibility
    """# solution 1: calculate upper triangle only (incomplete)
    # all combinations of elements to be compared, shape=len,2
    comb_vect = np.array(np.meshgrid(seg_mat, seg_mat)).T.reshape(-1,2)
    comb_vect_triup = np.unique(comb_vect.sort(), axis=0) # only compute upper triangle
    np.sum(np.sum(np.abs(np.diff(comb_vect_triup, axis=0)), axis=2), axis=1) 
    # TODO rebind vector to matrix..."""

    windows = seg_mat.shape[0]
    je_mat = np.zeros((windows, windows))

    for wy in range(windows):
        for wx in range(wy, windows):
            je_mat[wy, wx] = joint_entropy(seg_mat, wx, wy)

    # copies the upper triangle onto the lower one to have meaningful mean + std
    je_mat = je_mat + je_mat.T - np.diag(np.diag(je_mat)) 

    return je_mat


def differential_entropy(seg_mat):
    windows = seg_mat.shape[1]
    diff_mat = np.zeros(windows)

    for window in range(0, windows-1):
        diff_mat[window+1] = shannon_entropy(seg_mat[:,window+1]) - shannon_entropy(seg_mat[:,window])

    return diff_mat


def window_entropy(seg_mat):
    windows = seg_mat.shape[1]
    win_h_mat = np.zeros(windows)

    for window in range(0, windows):
        win_h_mat[window] = shannon_entropy(seg_mat[:,window])

    return win_h_mat


def mutual_information(je_mat):
    """Returns the mutual information of a matrix given a joint entropy matrix"""
    nrow = np.shape(je_mat)[0]
    mi_mat = np.zeros((nrow, nrow))

    for wy in range(nrow):
        for wx in range(wy, nrow):
            mi_mat[wy,wx] = je_mat[wy, wy] + je_mat[wx, wx] - je_mat[wy, wx]

    # copies the upper triangle onto the lower one to have meaningful mean + std
    mi_mat = mi_mat + mi_mat.T - np.diag(np.diag(mi_mat))

    return mi_mat


def transfer_entropy(pXY, pYY):
    """Returns the transfer entropy of two probability distributions"""
    #te = np.sum(pXY * np.log2(pXY / pYY)) # Schreiber et al. 2000

    # where p(YY) is 0, p(XY) is implied to be 0 and the contribution of the log term is interpreted as 0 
    pq_ratio = np.divide(pXY, pYY, out=np.zeros_like(pXY), where=pYY!=0)
    
    # result is zero if pXY is 0 because lim_x->0+ xlog(x) = 0
    te = np.sum(pXY * np.log2(pq_ratio, out=np.zeros_like(pq_ratio), where=pq_ratio!=0))

    return te


def all_transfer_entropy(trans_probs):
    """Returns the transfer entropy of all combinations of probability distributions"""
    # rolls the bin*bin nested array to the front 
    probs = np.moveaxis(trans_probs, (2,3), (0,1)) 
    # remove upper and lower triangular repr. long time intervals with k+1 history length as diagonal cutoff
    #probs = probs - np.triu(probs, k_history+1) - np.tril(probs, -k_history-1)
    bins = probs.shape[0]
    te_mat = np.zeros((bins, bins))

    for bin1 in range(0, bins):
        for bin2 in range(0, bins):
            #pYY = probs[bin2, bin2]
            
            # use the larger bin to avoid pYY actually being pXX
            pYY = probs[max(bin1, bin2), max(bin1, bin2)]
            pXY = probs[bin1, bin2]
            #print('pXY', pXY, pYY', pYY, sep='\n')
            te_mat[bin1, bin2] = transfer_entropy(pXY, pYY)

    return te_mat
            

def npmi_2d_fast(region1, region2):
    """Calculate an NPMI matrix between two regions
    Takes a list of :ref:`regions <regions>` and calculates the full NPMI
    matrix for all possible combinations of loci.
    This function uses vectorized numpy functions and is therefore much faster
    than npmi_2d_slow and is the recommended function for NPMI calculation.
    However, it is less readable and therefore potentially more difficult to
    debug.
    :param list regions: List of :ref:`regions <regions>`.
    :returns: :ref:`proximity matrix <proximity_matrices>` giving the npmi \
            of all possible combinations of windows within the different regions.
    """

    M = region1.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pXY = region1.dot(region2.T) / M
        pXpY = (region1.sum(1).reshape(-1, 1) / M) * (region2.sum(1) / M)
        pmi = np.log2(pXY / pXpY)
        npmi = pmi / -np.log2(pXY)
    return npmi


def pmi_2d_fast(region1, region2):

    M = region1.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pXY = region1.dot(region2.T) / M
        pXpY = (region1.sum(1).reshape(-1, 1) / M) * (region2.sum(1) / M)
        pmi = np.log2(pXY / pXpY)
    return pmi