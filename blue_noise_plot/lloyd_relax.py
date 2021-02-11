# -*- coding: utf-8 -*-
""" lloyd relaxtion for blue noise plots.
"""
import tensorflow as tf
import math
import copy
import numpy as np
from scipy.stats import gaussian_kde


def __compute_pdf(input_points, num_samples=8192, bandwidth=0.2):
    """ Compute a probability density function of a given set of points.

    Args:
        input_points (List[float]): 1D-Array of values
        num_samples (int): Number of samples for the pdf. Defaults to 8192.
        bandwidth (int): Bandwith for the kernel density estimation. This value is passed
            to `kde.factor`, used by `scipy.stats.gaussian_kde`. Defaults to 0.2.

    Returns:
        List[float]: A normalized (0-1) pdf of the input_points.
    """
    g_kde = gaussian_kde(dataset=input_points, bw_method=bandwidth)
    g_x = np.linspace(0, 1, num_samples)
    pdf = g_kde(g_x)
    return np.interp(pdf, (pdf.min(), pdf.max()), (0, 1))


def __jitter_sampler(N, scale_domain=1.0):
    sampler = np.zeros((N, 2))
    gridsize = int(np.sqrt(N))
    for x in range(gridsize):
        for y in range(gridsize):
            i = x * gridsize + y
            sampler[i][0] = (x + np.random.random_sample()) / gridsize
            sampler[i][1] = (y + np.random.random_sample()) / gridsize
    sampler[:, 0] *= scale_domain
    return sampler


def apply_jitter(data, aspect_ratio_scaling, centralized=False):
    """ Adds an additional dimension to the given data.

    Args:
        data (List[float]): List of data.
        aspect_ratio_scaling (float): Aspect ratio scaling, computed by using our adaptive height
            method.
        centralized (bool): Whether the samples should follow the density of the points in height
            as well to generate the samples in a violin plot like fashion.

    Returns:
       List[List[float]]: Data, given by X, with an additional (jittered) dimension.
    """
    point_count = len(data)

    jittered_x = np.zeros((data.size, 2))
    jittered_x[:, 0] = np.random.uniform(0, aspect_ratio_scaling, point_count)
    jittered_x[:, 1] = data

    if not centralized:
        return jittered_x

    new_jittered = []

    num_samples = 8192
    norm_pdf = __compute_pdf(data, num_samples)
    rejection_sampling_fn = (norm_pdf * (aspect_ratio_scaling / 2))

    for _, y in jittered_x:
        fn_value = np.interp(y, np.linspace(0, 1, num_samples),
                             rejection_sampling_fn)
        new_y = np.random.uniform(-fn_value, fn_value, 1)[0]
        new_jittered.append([new_y, y])

    new_jittered = np.array(new_jittered)
    new_jittered[:, 0] = new_jittered[:, 0] + (0.5 * aspect_ratio_scaling)
    return new_jittered


def __generate_adaptive_random_samples(input_points, aspect_ratio_scaling, centralized,
                                       num_samples=8182, bandwidth=0.2):
    """ Generates samples for Lloyd relaxation, following the density distribution of the input
    data.

    Args:
        input_points (List[float]): 1D-Array of values.: 1D-Array of values.
        scaling (float): Aspect ratio scaling, computed by using our adaptive height method.
        centralized (bool): Whether the samples should follow the density of the points in height
            as well to generate the samples in a violin plot like fashion.
        num_samples (int): Number of samples for the pdf. Defaults to 8192.
        bandwidth (int): Bandwith for the kernel density estimation. This value is passed
            to `kde.factor`, used by `scipy.stats.gaussian_kde`. Defaults to 0.2.

    Returns:
        List[List[float]] 2D-Array, of samples to be used for Lloyd relaxation.
    """
    norm_pdf = __compute_pdf(input_points, num_samples, bandwidth)

    if not centralized:
        y_grid = np.linspace(0, 1, num_samples)
        x = np.random.uniform(0, aspect_ratio_scaling, num_samples)
        y = np.random.choice(y_grid, size=num_samples,
                             p=norm_pdf / np.sum(norm_pdf))

        orig_sites = np.zeros((num_samples, 2))
        orig_sites[:, 0] = x
        orig_sites[:, 1] = y
        return orig_sites

    new_sites = []
    # I did offset the "rejection line" a bit from the center
    rejection_sampling_fn = (aspect_ratio_scaling / 10) + (norm_pdf * (aspect_ratio_scaling / 2))

    # I think until now, we had 2 * num_samples. because num_samples for
    # each axis. This ensures, we have at least this number... should be
    # updated to something more useful.
    while len(new_sites) < num_samples:
        y_values = np.random.uniform(0, 1, num_samples)
        # generating data samples "around" the 0-line
        x_values = np.random.uniform(-0.5 * aspect_ratio_scaling,
                                     0.5 * aspect_ratio_scaling,
                                     num_samples)
        orig_sites = np.zeros((num_samples, 2))
        orig_sites[:, 0] = x_values
        orig_sites[:, 1] = y_values

        for x, y in orig_sites:
            fn_value = np.interp(y, np.linspace(0, 1, num_samples),
                                 rejection_sampling_fn)
            # if point is within the function, take it, otherwise reject.
            if np.abs(x) < fn_value:
                new_sites.append([x, y])

    new_sites = np.array(new_sites)
    new_sites = new_sites[:num_samples]
    # move the sampling points back to the original place
    new_sites[:, 0] = new_sites[:, 0] + (0.5 * aspect_ratio_scaling)

    return new_sites


def __compute_voronoi_regions(points_per_site, points):
    """ Generates samples for Lloyd relaxation, following the density distribution of the input
    data.

    Args:
        input_points (List[float]): 1D-Array of values.: 1D-Array of values.
        scaling (float): Aspect ratio scaling, computed by using our adaptive height method.
        centralized (bool): Whether the samples should follow the density of the points in height
            as well to generate the samples in a violin plot like fashion.
        num_samples (int): Number of samples for the pdf. Defaults to 8192.
        bandwidth (int): Bandwith for the kernel density estimation. This value is passed
            to `kde.factor`, used by `scipy.stats.gaussian_kde`. Defaults to 0.2.

    Returns:
        List[List[float]] 2D-Array, of samples to be used for Lloyd relaxation.
    """
    diff = tf.math.subtract(points_per_site, points)
    dist = tf.norm(tf.math.multiply(diff, tf.constant([2, 1], dtype=tf.float32)),
                   ord=1, axis=2, keepdims=True)
    return tf.math.argmin(dist, axis=1)


def __compute_centroids(voronoi, points_index_tensor, sites_per_point, ones, zeros, num_points,
                        num_sites, num_dims_per_points):
    # compute centroids
    mask = tf.math.equal(points_index_tensor, voronoi)
    mask_tiles = tf.tile(mask, [1, 1, num_dims_per_points])
    sites_mask = tf.reshape(mask_tiles, [num_points, num_sites, num_dims_per_points])
    masked_sites_sum = tf.squeeze(tf.math.reduce_sum(tf.where(sites_mask, sites_per_point, zeros),
                                                     axis=1, keepdims=True))
    counts = tf.split(tf.squeeze(tf.reduce_sum(tf.where(sites_mask, ones, zeros),
                                               axis=1, keepdims=True)),
                      num_dims_per_points, axis=1)[0]
    return tf.math.divide(masked_sites_sum, counts)


def blue_noise_single_class(input_points, aspect_ratio_scaling,
                            centralized=False, num_samples_per_dim=100 ** 2,
                            max_iterations=200):
    points = apply_jitter(input_points, aspect_ratio_scaling,
                          centralized)
    sites = __generate_adaptive_random_samples(input_points,
                                               aspect_ratio_scaling,
                                               centralized,
                                               num_samples_per_dim)

    points = tf.convert_to_tensor(points, dtype=tf.float32)
    sites = tf.convert_to_tensor(sites, dtype=tf.float32)

    original_values = tf.split(points, num_or_size_splits=2, axis=1)[1]

    num_points = tf.shape(points).numpy()[0]
    num_dims_per_points = tf.shape(points).numpy()[1]
    num_sites = tf.shape(sites).numpy()[0]

    points_per_site = tf.reshape(tf.tile(sites, [1, num_points]),
                                 [num_sites, num_points, num_dims_per_points])
    sites_per_point = tf.reshape(tf.tile(sites, [num_points, 1]),
                                 [num_points, num_sites, num_dims_per_points])

    zeros = tf.zeros(sites_per_point.shape);
    ones = tf.ones(sites_per_point.shape);

    points_index_tensor = [x for x in range(num_points)]
    points_index_tensor = tf.expand_dims(tf.convert_to_tensor(points_index_tensor, dtype=tf.int64),
                                         axis=1)
    points_index_tensor = tf.expand_dims(tf.tile(points_index_tensor, [1, num_sites]),
                                         axis=2)

    # Lloyd iterations
    for i in range(max_iterations):
        voronoi = __compute_voronoi_regions(points_per_site, points)

        centroids = __compute_centroids(voronoi, points_index_tensor, sites_per_point, ones, zeros,
                                        num_points, num_sites, num_dims_per_points)

        x, y = tf.split(centroids, num_or_size_splits=2, axis=1)
        isOutsidePlot = tf.math.is_nan(x)
        correctedX = tf.where(isOutsidePlot, tf.constant([aspect_ratio_scaling / 2.0], dtype=tf.float32), x)

        # put the relax point, back to it's original data-dimension.
        points = tf.squeeze(tf.stack([correctedX, original_values], axis=1))

    return points.numpy()


def mc_compute_centroids(samples, voronoi, original_y):
    num_points = len(original_y)
    num_samples = len(samples)

    centroids = np.zeros((num_points, 2))
    counter = np.zeros(num_points)

    for p in range(num_samples):
        siteIndex = int(voronoi[p])
        centroids[siteIndex] += samples[p]
        counter[siteIndex] += 1

    centroids = np.array([np.divide(centroid,
                                    count,
                                    out=np.zeros_like(centroid),
                                    where=count != 0)
                          for centroid, count in zip(centroids, counter)])

    # To keep the y-coordinate unchanged, revert it back to the initial value
    centroids[:, 1] = original_y
    return centroids


def mc_get_closest_site_index(sample, sites):
    reshaped_samples = np.tile(sample, (sites.shape[0], 1))
    diff = sites - reshaped_samples
    return np.argmin((2 * np.fabs(diff[:, 0])) + np.fabs(diff[:, 1]))


mc_compute_voronoi_regions = np.vectorize(mc_get_closest_site_index,
                                         signature='(n),(m,n)->()')


def blue_noise_multi_class(data, aspect_ratio_scaling, centralized,
                           num_samples_per_dim=8192, max_iterations=100,
                           iteration_step_cb=None):
    jittered_points = apply_jitter(data['points'], aspect_ratio_scaling,
                                   centralized)
    sites = __generate_adaptive_random_samples(data['points'],
                                               aspect_ratio_scaling,
                                               centralized,
                                               num_samples_per_dim)

    points = copy.deepcopy(jittered_points)
    voronoi = np.zeros(len(sites))

    # Lloyd iterations
    for itr in range(max_iterations):
        start_index = 0
        class_centroids = np.zeros((jittered_points.shape))
        # Loop over all the classes in the dataset
        for i in range(len(data['points_per_class'])):
            num_elements = data['points_per_class'][i]
            end_index = start_index + num_elements
            elements = points[start_index:end_index]

            class_voronoi = mc_compute_voronoi_regions(sites, elements)

            # update the sites to the centroids of their voronoi regions of the i-th  class
            class_centroids[start_index:end_index] = mc_compute_centroids(sites,
                                                                          class_voronoi,
                                                                          elements[:, 1])
            # iteration_step_cb(itr, voronoi, original_samples, elements, rgb)
            start_index += num_elements

        # class-wise centroids are computed, replace points with that
        points = class_centroids

        # compute the voronoi for all the sites
        voronoi = mc_compute_voronoi_regions(sites, points)

        # update all the sites to the centroids of their voronoi regions
        centroids = mc_compute_centroids(sites, voronoi, data['points'])

        points = centroids

    return points
