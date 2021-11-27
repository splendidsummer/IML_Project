import os
import numbers
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split


def save_fig(fig_path, tight_layout=True, fig_extension="jpg", resolution=300):
    path = os.path.join( fig_path + "." + fig_extension)
    print("Saving figure", fig_path)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def matching_dissim(a, b, **_):
    """Simple matching dissimilarity function"""
    return np.sum(a != b, axis=1)


def get_max_value_key(dic):
    """Gets the key for the maximum value in a dict."""
    v = np.array(list(dic.values()))
    k = np.array(list(dic.keys()))

    maxima = np.where(v == np.max(v))[0]
    if len(maxima) == 1:
        return k[maxima[0]]
    else:
        # In order to be consistent, always selects the minimum key
        # (guaranteed to be unique) when there are multiple maximum values.
        return k[maxima[np.argmin(k[maxima])]]


def move_point_cat(point, ipoint, to_clust, from_clust, cl_attr_freq,
                   membship, centroids):
    """Move point between clusters, categorical attributes."""
    membship[to_clust, ipoint] = 1
    membship[from_clust, ipoint] = 0
    # Update frequencies of attributes in cluster.
    for iattr, curattr in enumerate(point):
        to_attr_counts = cl_attr_freq[to_clust][iattr]
        from_attr_counts = cl_attr_freq[from_clust][iattr]

        # Increment the attribute count for the new "to" cluster
        to_attr_counts[curattr] += 1

        current_attribute_value_freq = to_attr_counts[curattr]
        current_centroid_value = centroids[to_clust][iattr]
        current_centroid_freq = to_attr_counts[current_centroid_value]
        if current_centroid_freq < current_attribute_value_freq:
            # We have incremented this value to the new mode. Update the centroid.
            centroids[to_clust][iattr] = curattr

        # Decrement the attribute count for the old "from" cluster
        from_attr_counts[curattr] -= 1

        old_centroid_value = centroids[from_clust][iattr]
        if old_centroid_value == curattr:
            # We have just removed a count from the old centroid value. We need to
            # recalculate the centroid as it may no longer be the maximum
            centroids[from_clust][iattr] = get_max_value_key(from_attr_counts)

    return cl_attr_freq, membship, centroids


def load_split_data(name='', test_rate=0.25):
    """
    load dataset
    :param name: which data to train
    :param test_rate: the test rate of all data
    :return:
    """
    data_root = 'dataset'
    if name == 'pen-based':
        print('------------------ the cur data set is pen ----------------')
        data_path = os.path.join(data_root, 'pen-based.pkl')
        label_path = os.path.join(data_root, 'pen-based_label.pkl')
    elif name == 'satimage':
        print('------------------ the cur data set is sat ----------------')
        data_path = os.path.join(data_root, 'satimage.pkl')
        label_path = os.path.join(data_root, 'satimage_label.pkl')
    data = pickle.load(open(data_path, 'rb'), encoding='utf-8')
    label = pickle.load(open(label_path, 'rb'), encoding='utf-8')

    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=test_rate, stratify=label)
    return data_train, data_test, label_train, label_test


def load_data(name=''):
    """
    load dataset
    :param name: which data to train
    :param test_rate: the test rate of all data
    :return:
    """
    data_root = 'dataset'
    if name == 'pen-based':
        print('------------------ the cur data set is pen ----------------')
        data_path = os.path.join(data_root, 'pen-based.pkl')
        label_path = os.path.join(data_root, 'pen-based_label.pkl')
    elif name == 'satimage':
        print('------------------ the cur data set is sat ----------------')
        data_path = os.path.join(data_root, 'satimage.pkl')
        label_path = os.path.join(data_root, 'satimage_label.pkl')
    data = pickle.load(open(data_path, 'rb'), encoding='utf-8')
    labels = pickle.load(open(label_path, 'rb'), encoding='utf-8')

    return data, labels


# def df_to_array(x):
#     return x.values if 'pandas' in str(x.__class__) else x
#
#     def labels_cost(X, centroids, dissim):
#         """Calculate labels and cost function given a matrix of points and
#         a list of centroids for the k-modes algorithm.
#         """
#     n_points = X.shape[0]
#     cost = 0.
#     labels = np.empty(n_points, dtype=np.uint16)
#     for ipoint, curpoint in enumerate(X):
#         diss = dissim(centroids, curpoint)
#         clust = np.argmin(diss)
#         labels[ipoint] = clust
#         cost += diss[clust]
#
#     return labels, cost



