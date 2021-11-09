"""
K-modes clustering for categorical data
"""

from collections import defaultdict
import random
import numpy as np
import pickle
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array

from utils.utils_function import labels_cost, move_point_cat, get_max_value_key


def k_modes(X, n_clusters, max_iteration, dissim, single_seed, random=False):
    """

    :param n_clusters:
    :param dissim:
    :param single_seed:
    :param random:
    :return:
    """

    random.seed(single_seed)
    np.random.seed(single_seed)
    n_points, n_attrs = X.shape

    if not random:
        centroids = np.zeros((n_clusters, n_attrs))
        max_states = X.max(axis=0)

        for max_state in max_states:
            idx = 0
            random_centroid_seeds = np.random.randint(1, max_state + 1, size=n_clusters)
            centroids[:, idx] = random_centroid_seeds
            idx += 1

    else:
        seeds = np.random.choice(range(n_points), n_clusters)
        centroids = X[seeds]

    # random_state = np.random.mtrand._rand
    # random_seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)

    if sparse.issparse(X):
        raise TypeError("k-modes does not support sparse data.")

    assert n_clusters <= n_points, "Cannot have more clusters ({}) " \
                                   "than data points ({}).".format(n_clusters, n_points)

    membship = np.zeros((n_clusters, n_points), dtype=np.uint8)

    # cl_attr_freq is a list of lists with dictionaries that contain the
    # frequencies of values per cluster and attribute.

    cl_attr_freq = [[defaultdict(int) for _ in range(n_attrs)]
                    for _ in range(n_clusters)]

    for ipoint, curpoint in enumerate(X):
        # Initial assignment to clusters
        clust = np.argmin(dissim(centroids, curpoint, X=X, membship=membship))
        membship[clust, ipoint] = 1
        # Count attribute values per cluster.
        for iattr, curattr in enumerate(curpoint):
            cl_attr_freq[clust][iattr][curattr] += 1
    # Perform an initial centroid update.
    for ik in range(n_clusters):
        for iattr in range(n_attrs):
            if sum(membship[ik]) == 0:
                # Empty centroid, choose randomly
                centroids[ik, iattr] = np.random.choice(X[:, iattr])
            else:
                centroids[ik, iattr] = get_max_value_key(cl_attr_freq[ik][iattr])

    iteration = 0
    labels = None
    converged = False

    _, cost = labels_cost(X, centroids, dissim)
    epoch_costs = [cost]

    while iteration < max_iteration and not converged:
        iteration += 1
        centroids, moves = k_modes_one_iter(X, centroids, cl_attr_freq, membship, dissim)
        labels, ncost = labels_cost(X, centroids, dissim, membship)
        converged = (moves == 0) or (ncost >= cost)
        epoch_costs.append(ncost)
        cost = ncost
        print("Run iteration: {}/{}, moves: {}, cost: {}".format(iteration, max_iteration, moves, cost))

    return centroids, labels, cost, iteration, epoch_costs


def k_modes_one_iter(X, centroids, cl_attr_freq, membship, dissim):

    moves = 0.

    for ipoint, curpoint in enumerate(X):
        clust = np.argmin(dissim(centroids, curpoint))
        if membship[clust, ipoint]:
            # Point is already in its right place.
            continue

        moves += 1
        old_clust = np.argwhere(membship[:, ipoint])[0][0]

        cl_attr_freq, membship, centroids = move_point_cat(
            curpoint, ipoint, clust, old_clust, cl_attr_freq, membship, centroids
        )

        # In case of an empty cluster, reinitialize with a random point
        # from the largest cluster.
        if not membship[old_clust, :].any():
            from_clust = membship.sum(axis=1).argmax()
            choices = [ii for ii, ch in enumerate(membship[from_clust, :]) if ch]
            rindx = np.random.choice(choices)

            cl_attr_freq, membship, centroids = move_point_cat(
                X[rindx], rindx, old_clust, from_clust, cl_attr_freq, membship, centroids
            )

    return centroids, moves


