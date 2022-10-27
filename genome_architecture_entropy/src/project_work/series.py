"""
Title: Series
Authors: René Schmiedler
Date: 20.08.2021

Description 
-----------
This file allows the user to generate a configuration space using a series of realizations with a
start and end state between which a weighted mean and noise are applied.
It can be imported as a module.

Requires
--------
* numpy
* shapely

Functions
---------
* series_conf_space - generates an ensemble of specified length between a given star and end state
* generate_series   - computes a series of states according to transition probabilities
"""

import numpy as np

import toymodel.sampling as cs
import plotting as plot


def series_conf_space(conf_length, start_coords, end_coords):
    """
    Generates an ensemble of specified length between a given star and end state. 
    
    Each in between realization is a weighted mean between start and end with noise added and sliced
    into a segregation matrix.
    
    Arguments
    ---------
    conf_length : int
        desired number of realizations
    start_coords : numpy array
        two row vectors storing x and y coordinates respectively
    end_coords : numpy array
        two row vectors storing x and y coordinates respectively
        
    Return
    ------
    seg_mats : numpy array
        all segregation matrices stacked vertically
    """

    states = np.array([start_coords, end_coords])
    realizations = np.empty((conf_length, 2, states.shape[2]))
    seg_mats = np.empty((conf_length, 1000, states.shape[2]))

    for t in range(0, conf_length):
        scale = t / (conf_length-1)
        weight = np.full_like(states[0], scale)
        weights = np.array([1-weight, weight])
        realizations[t] = np.average(states, axis=0, weights=weights)
        realizations[t] = cs.add_noise(realizations[t], 0.08)
        seg_mats[t] = cs.slice(realizations[t], 0, 36, -5, 5)
        #plot.plot_xy(realizations[t], 0,35,-10,10)

    return seg_mats


def generate_series(seg_mat, length, prnt=False):
    """
    Computes a series of states according to transition probabilities.
    
    This function evaluates the similarity of all pairs of states and calculates a normalized 
    distance. A random walk of a time-homogenous markov chain in a finite state space is 
    computed (= a series of realizations) according to transition probabilities derived from 
    the distance measure.
    
    Arguments
    ---------
    seg_mat : numpy array
        segregation matrices of the configuration space stacked vertically in the
    length : int
        length of series to be generated
    prnt : bool, optional
        A flag used to print additional information (default is False)

    Return
    ------
    series : numpy vector of int
        vector of references to realizations in the configuration space
    """

    # element-wise differences of all pairs of vectors 
    all_dist_vect = seg_mat[:, None] - seg_mat[None, :] 
    
    # summed differences of all pairs of vectors in symmetrical matrix
    # this is equivalent to an adjacency matrix of a maximum clique graph with weights, no loops/multiedges
    dist_mat = np.sum(np.sum(np.abs(all_dist_vect), axis=-1), axis=-1) 

    # scale the distances so that the min=0 and max=1
    #dist_temp = dist_mat - np.min(dist_mat[dist_mat!=0])
    #dist_mat = dist_temp / np.max(dist_temp)

    # set self, first and last realization as not reachable
    #np.fill_diagonal(dist_mat, 0)
    dist_mat[:,0] = 0
    dist_mat[:,-1] = 0 

    outdeg = np.sum(dist_mat, axis=1)[:, None] # weighted degree
    deg_mat = np.zeros_like(dist_mat) # degree matrix
    np.fill_diagonal(deg_mat, outdeg)
    deg_mat = np.linalg.inv(deg_mat)

    prob_mat = np.matmul(deg_mat, dist_mat) # walk or diffusion matrix
    
    ncol = np.shape(dist_mat)[1]
    x_mat = np.zeros(shape=(length, ncol)) 
    series = np.zeros(shape=length)
    
    #xt = np.matmul(x_mat[t-1,:], prob_mat) # next timestep

    for t in range(0, length):
        if t==0: 
            # start with 100% chance of starting at realization 1
            x_mat[0,0] = 1 
            series[0] = 0
        elif t==length-1:
            # last realization in sequence is always end state
            series[t] = prob_mat.shape[1]-1 
        else: 
            last = series[t-1].astype(int)
            # pick a col according to probs in row of last realization (walk traversal)
            series[t] = np.random.choice(np.where(prob_mat[last] == np.random.choice(prob_mat[last], 1, p=prob_mat[last]))[0])

    if prnt:
        #print('\n degrees: \n', deg_mat, '\n')
        print('Distance matrix:\n', dist_mat, '\n')
        print('Probability of walk matrix:\n', prob_mat, '\n')
        #print('Transition probabilies:\n', x_mat, '\n')
        print('x(t) sequence of realizations:\n', series+1)
        #print('x(t), sequence of distances:\n', x[1:,], '\n')
    
    return series
