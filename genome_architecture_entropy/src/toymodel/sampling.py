"""
Title: Configuration Space
Authors: René Schmiedler
Date: 20.08.2021

Description 
-----------
This file allows the user to manipulate xy coordinates and then sample the data using an intersecting
slice as well as bind together mutliple data sets. It can be imported as a module.

Requires
--------
* numpy
* shapely

Functions
---------
* rotate         - rotates a set of xy coordinates to a degree around a given origin
* add_noise      - adds a specified amount of noise to a set of xy coordinates
* slice          - intersects a set of xy coordinates and records for each point the slices it
                    intersects with
* ran_conf_space - generates variaions of a model by applying noise and rotation
"""

import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shapely.affinity as sa
from shapely.geometry import MultiPoint, box
matplotlib.rcParams["figure.dpi"] = 300

def rotate(xy_coord_mat, degree, origin=(5, 5)):
    """This function rotates a set of xy coordinates to a degree around a given origin."""
    radians = np.radians(degree)
    x, y = xy_coord_mat
    ox, oy = origin
    qx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
    qy = oy + -math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)

    return np.array([qx, qy])


def add_noise(xy_coord_mat, amount):
    """This function adds a specified amount of noise to a set of xy coordinates."""
    noise = np.random.normal(0, amount, xy_coord_mat.shape)
    noisy_signal = xy_coord_mat + noise

    return noisy_signal


def slice(xy_coord_mat, sample_n=1000, wdf_tresholds=[0.1, 0.3]):
    """
    This function intersects a set of xy coordinates and records for each point the slices it
    intersects with. 
    
    The xy-plane is intersected 1000 times by a rectangular shape, a slice, covering the maximum 
    width of the plane at a height of 5% in a random orientation. The position of the slice is 
    normally distributed in order to simulate the slicing of an approximately spherical nucleus, 
    where the center is statistically more likely to be sampled
    
    Arguments
    ----------
    xy_coord_mat : numpy array
        two row vectors storing x and y coordinates respectively
    xmin : float
    xmax : float 
    ymin : float 
    ymax : float 
        dimensions of the coordinate system
        
    Return
    ------
    seg_mat : numpy array
        segregation matrix of all points contained within a slice
    """
    plot = False 

    xmin_slice = np.min(xy_coord_mat[0])
    xmax_slice = np.max(xy_coord_mat[0])
    ymin_slice = np.min(xy_coord_mat[1])
    ymax_slice = np.max(xy_coord_mat[1])

    width_slice = xmax_slice - xmin_slice
    height_slice = ymax_slice - ymin_slice

    # larger area to escape boundary effects
    # sample > slice
    scale = 1
    xmin_sample = xmin_slice - scale * width_slice    
    xmax_sample = xmax_slice + scale * width_slice
    ymin_sample = ymin_slice - scale * height_slice
    ymax_sample = ymax_slice + scale * height_slice

    width_sample = xmax_sample - xmin_sample
    height_sample = ymax_sample - ymin_sample

    sample_center = (xmax_sample + xmin_sample) / 2
    sample_xmin = sample_center - 3 * width_slice
    sample_xmax = sample_center + 3 * width_slice
    sample_height = (ymax_sample + np.abs(ymin_sample)) / 100 * 5 

    points  = MultiPoint(np.stack(xy_coord_mat, axis=1))
    seg_mat = np.zeros((sample_n, len(points.geoms)))
    segregation = np.zeros(len(points.geoms))
    n_loci = xy_coord_mat.shape[1]
    wdf_tresholds = [int(treshold * n_loci) for treshold in wdf_tresholds] # window detection frequency treshold (lower bound)

    #for iter in range(sample_n):
    iter = 0
    while iter < sample_n:
        rand_y     = np.random.uniform(low=ymin_sample, high=ymax_sample)
        slice_ymin = rand_y - 0.5 * sample_height
        slice_ymax = rand_y + 0.5 * sample_height
        slice_      = box(sample_xmin, slice_ymin, sample_xmax, slice_ymax)

        rand_x     = np.random.uniform(low=xmin_sample, high=xmax_sample)
        rand_angle = np.random.uniform(-180, 180)
        slice_      = sa.rotate(slice_, rand_angle, origin=(rand_x, rand_y))
        x,y        = slice_.exterior.xy

        for p in range(len(points.geoms)):
            segregation[p] = slice_.contains(points[p])
            #seg_mat[iter, p] = slice_.contains(points[p])

        # next iteration only if slice is non-empty (any True)
        if np.sum(segregation) > wdf_tresholds[0] and np.sum(segregation) < wdf_tresholds[1]:
            # check if slice is a duplicate (no new information)
            #if not any(np.equal(seg_mat, segregation).all(1)):
            seg_mat[iter] = segregation
            #if iter % 50 == 0: print(iter)
            iter = iter + 1
            

            if plot:
                plt.plot(x,y, linewidth = 0.1, color='k', fillstyle='full', alpha=0.01)
                ax = plt.gca()
                ax.set_xlim([xmin_sample - width_sample, xmax_sample + width_sample])
                ax.set_ylim([ymin_sample - height_sample, ymax_sample + height_sample])
                ax.set_aspect('equal', 'box')

    if plot:
        sample_area = patches.Rectangle((xmin_sample, ymin_sample), 
                                        width_sample, height_sample, 
                                        linewidth=.5, 
                                        edgecolor='white', 
                                        facecolor='none', 
                                        zorder=sample_n)

        slice_area = patches.Rectangle((xmin_slice, ymin_slice), 
                                        width_slice, height_slice, 
                                        linewidth=.5, 
                                        edgecolor='red', 
                                        facecolor='none', 
                                        zorder=sample_n)
        ax.add_patch(slice_area)
        ax.add_patch(sample_area)
        plt.show()

    return seg_mat


def ran_conf_space(conf_length, model, deg_range=(0, 181), noise=0.1):
    """This function generates variaions of a model by applying noise and rotation."""
    seg_mats = np.empty((conf_length, 1000, model.shape[1]))

    for conf in range(0, conf_length):
        mod_model = add_noise(model, noise)
        mod_model = rotate(mod_model, np.random.uniform(*deg_range))
        seg_mats[conf,:,:] = slice(mod_model, 0, 10, 0, 10)

    return seg_mats
