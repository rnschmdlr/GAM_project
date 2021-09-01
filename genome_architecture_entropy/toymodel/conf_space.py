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
import shapely.affinity as sa
from shapely.geometry import MultiPoint, box


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


def slice(xy_coord_mat, minx, maxx, miny, maxy, sample_n=1000):
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
    minx : float
    maxx : float 
    miny : float 
    maxy : float 
        dimensions of the coordinate system
        
    Return
    ------
    seg_mat : numpy array
        segregation matrix of all points contained within a slice
    """

    points  = MultiPoint(np.stack(xy_coord_mat, axis=1))
    seg_mat = np.zeros((sample_n, len(points.geoms)))

    loc = (maxy + miny) / 2
    
    slice_center = (maxx + minx) / 2
    slice_maxx = slice_center + maxx * 2
    slice_minx = slice_center - maxx * 2
    slice_height = (maxy + np.abs(miny)) / 100 * 5

    for iter in range(sample_n):
        rand_angle  = np.random.uniform(0, 181)
        rand_y      = np.random.normal(loc, scale=1)
        slice       = box(slice_minx, 
                          rand_y - 0.5* slice_height, 
                          slice_maxx, 
                          rand_y + 0.5 * slice_height)
        slice       = sa.rotate(slice, rand_angle, origin='center')
        #x,y        = slice.exterior.xy
        #plt.plot(x,y)

        for p in range(len(points.geoms)):
            seg_mat[iter, p] = slice.contains(points[p])

    return seg_mat


def ran_conf_space(conf_length, model, deg_range=(0, 181), noise=0.1):
    """This function generates variaions of a model by applying noise and rotation."""
    seg_mats = np.empty((conf_length, 1000, model.shape[1]))

    for conf in range(0, conf_length):
        mod_model = add_noise(model, noise)
        mod_model = rotate(mod_model, np.random.uniform(*deg_range))
        seg_mats[conf,:,:] = slice(mod_model, 0, 10, 0, 10)

    return seg_mats
