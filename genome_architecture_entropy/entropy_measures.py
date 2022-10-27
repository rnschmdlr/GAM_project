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
* shannon_entropy          - returns total entropy of a matrix
* shannon_entropy_multivar - returns the entropy for each variable (row or col) of a matrix
* joint_entropy            - returns joint entropy of two variables in a matrix
* all_joint_entropy        - returns joint entropy of all combination of row vectors in a matrix
* mutual_information       - returns mutual information of a matrix given a joint entropy matrix
* transfer_entropy         - returns transfer entropy of two probability distributions 
* all_transfer_entropy     - returns transfer entropy of all combinations of probability distributions
"""

import numpy as np
from scipy.stats import entropy
import warnings
    

def shannon_entropy(seg_mat):
    """Returns the total entropy of a matrix"""
    freq = np.unique(seg_mat, return_counts=True)[1] / np.prod(seg_mat.shape)
    h = -np.sum(freq * np.log2(freq, out=np.zeros_like(freq), where=freq!=0))
    
    return h

def corrected_shannon_entropy(matrix):
    h = shannon_entropy(matrix)
    corr_h = 1 - np.sqrt(1 - h ** 4/3)

    return corr_h


def shannon_entropy_multivar(matrix, axis):
    """Returns the entropy for each variable (row or col) of a matrix"""
    ones = np.apply_along_axis(np.sum, axis, matrix)
    zeros = matrix.shape[axis] - ones
    counts = np.vstack([ones, zeros]).T
    h = entropy(counts, base=2, axis=axis)
    
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


def normalized_mutual_information(seg_mat, c=True):
    je_mat = all_joint_entropy(seg_mat)
    mi_mat = mutual_information(je_mat)
    nloci = np.shape(mi_mat)[0]
    nmi_mat = np.zeros_like(mi_mat)
    entropy_list = shannon_entropy_multivar(seg_mat, 1)

    for wy in range(nloci):
        for wx in range(nloci):
            hx = entropy_list[wx]
            hy = entropy_list[wy]
            norm = np.max([hx, hy])
            # normalization
            nmi_mat[wy,wx] = np.divide(mi_mat[wy,wx], norm, out=np.zeros_like(mi_mat[wy,wx]), where=norm!=0)
            # corrected nmi
            if c:
                nmi_mat[wy,wx] = 1 - (1 - np.sqrt(nmi_mat[wy,wx]))**1.2315

    return nmi_mat


def transfer_entropy(pXY, pyYX, pyY):
    """Returns the transfer entropy of two probability distributions"""
    # Schreiber et al. 2000

    # where p(YY) is 0, p(XY) is implied to be 0 and the contribution of the log term is interpreted as 0 
    pq_ratio = np.divide(pyYX, pyY, out=np.zeros_like(pyYX), where=pyY!=0)

    # result is zero if pXY is 0 because lim_x->0+ xlog(x) = 0
    te = np.sum(pXY * np.sum(pyYX * np.log2(pq_ratio, out=np.zeros_like(pq_ratio), where=pq_ratio!=0), axis=1), axis=0)

    return te


def multivariate_transfer_entropy_(transition_pmat):
    n_loci = transition_pmat.shape[0]
    n_tsteps = transition_pmat.shape[2]
    te_mat = np.zeros(shape=(n_loci, n_loci))

    for loc1 in range(n_loci):
        YY = transition_pmat[loc1, loc1, :, :]
        for loc2 in range(n_loci):
            XY = transition_pmat[loc1, loc2, :, :]

            P = transition_pmat[loc1, loc2]
            p = 1
            p1 = np.linalg.matrix_power(P, p)
            p2 = np.linalg.matrix_power(P, p+1)
            # find the stationary distribution by increasing the power until no further change
            while not np.allclose(p1,p2) and not (p1==0).all() and not p > 100: 
                p += 1
                p1 = np.linalg.matrix_power(p1, p)
                p2 = np.linalg.matrix_power(p1, p+1)
            p1 = np.ones((n_tsteps))
            te_mat[loc1,loc2] = transfer_entropy(p1, XY, YY)

    return te_mat


def multivariate_transfer_entropy(transition_pmat):
    """Returns the transfer entropy of all combinations of probability distributions"""
    # compute stationary distribution over all states by increasing the power until no further change
    n_loci = transition_pmat.shape[2]
    n_tsteps = transition_pmat.shape[0]
    stat_dist = np.zeros(shape=(n_tsteps, n_tsteps, n_loci, n_loci))

    for t1 in range(n_tsteps):
        for t2 in range(n_tsteps):
            P = transition_pmat[t1,t2]
            p = 1
            p1 = np.linalg.matrix_power(P, p)
            p2 = np.linalg.matrix_power(P, p+1)
            # find the stationary distribution by increasing the power until no further change
            while not np.allclose(p1,p2) and not (p1==0).all() and not p > 100: 
                p += 1
                p1 = np.linalg.matrix_power(p1, p)
                p2 = np.linalg.matrix_power(p1, p+1)
            stat_dist[t1,t2] = p1

            # 
            #YY = np.diag(transition_pmat[t1, t2])
            #XY = np.sum(transition_pmat[t1,t2], axis=1) 

            #for loc1 in range(n_loci):
            #    #YY = np.sum(transition_pmat[t1, t2, loc1, loc1], axis=0)
            #    for loc2 in range(n_loci):
            #        #XY = np.sum(transition_pmat[t1,t2, loc1, loc2])
            #        te_mat[loc1,loc2] = transfer_entropy(stat_dist[t1,t2], XY, YY)

    # rolls the loci bin*bin nested array to the front 
    transition_pmat_inner = np.moveaxis(transition_pmat, (2,3), (0,1)) 
    stat_dist_inner = np.moveaxis(stat_dist, (2,3), (0,1))
    te_mat = np.zeros(shape=(n_loci, n_loci))

    for loc1 in range(n_loci):
        yY = transition_pmat_inner[loc1, loc1]

        for loc2 in range(n_loci):
            yXY = transition_pmat_inner[loc1, loc2]

            #pyY = np.sum(yY, axis=0, dtype=np.float64)
            #pyXY = np.sum(yXY, axis=0, dtype=np.float64)
            #pXY = np.ones_like(pyXY)
            pXY = stat_dist_inner[loc1, loc2]
            pXY = np.ones((n_tsteps))
            te_mat[loc1, loc2] = transfer_entropy(pXY, yXY, yY)

    #print(t1, t2, '\n')
    #print('p1: ', p1)
    #print(YY, YY.shape, '\n')
    #print(XY, XY.shape, '\n')
    #print(stat_dist[t1,t2], stat_dist[t1,t2].shape, '\n')
   
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