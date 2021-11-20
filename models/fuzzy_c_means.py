import copy
import math
import random
import time
import numpy as np
import pandas as pd
from fcmeans import FCM


class fuzzyCMeans:

    def __init__(self, k_clusters, data_arr, random_state, max_iter=10000, m=2, epsilon=1e-5):

        self.k_clusters = k_clusters
        self.data = data_arr
        self.n_samples = data_arr.shape[0]
        self.epsilon = epsilon
        self.gold_standard = data_arr[:, -1]
        self.membership_matrix = None
        self.centroids = None
        self.epsilon = epsilon
        self.random_generator = np.random.default_rng(random_state)
        self.m = m
        self.max_iter = max_iter

    def initialize_membership_matrix(self):

        self.membership_matrix = self.random_generator.uniform(size=(self.n_samples, self.k_clusters))
        membership_normalizer = self.membership_matrix.sum(axis=1)[np.newaxis].T
        membership_normalizer = np.tile(membership_normalizer, self.k_clusters)

        self.membership_matrix = self.membership_matrix / membership_normalizer

    def find_centroids(self):
        """Update cluster centers"""
        u_power_m = self.membership_matrix ** self.m
        numerator = np.dot(self.data.T, u_power_m)
        denominator = np.sum(u_power_m, axis=0)  # sum the u_ij values of all samples in cluster i

        self.centroids = (numerator/denominator).T

    def update_membership_matrix(self):
        """ Compute the new membership or U matrix
         Parameters
        ----------
        self.data: array, shape = [n_samples, n_features], training data.
        self.centroids: array, shape [k_clusters, n_features], centroids in current iteration

        Returns
        -------
        array, shape = [n_samples, n_clusters], membership function
            Fuzzy partition array, returned as an array with n_samples rows and n_clusters columns.
        """
        dist_samples_centroid = self._compute_distance()
        dist_samples_centroid = dist_samples_centroid ** float(2/(self.m-1))
        reshape_dist = dist_samples_centroid.reshape((self.data.shape[0], 1, -1))  # add a new dimension in the middle
        # expand the second dimension from 1 to
        dist_on_all_centroids = reshape_dist.repeat(reshape_dist.shape[-1], axis=1)
        denominator = dist_samples_centroid[:, :, np.newaxis] / dist_on_all_centroids

        return 1 / denominator.sum(2)

    def _compute_distance(self):
        """  Using "_compute_distance" to calculate the distance between sample and centroid
               1.  Firstly expand an extra dimension for  training samples by self.data[:, None, :]
                   or self.data[:, np.newaxis, :]
               2.  Get the distance between a certain sample and a certain centroid by using np.einsum to get sum along
                   the feature dimension.
                   return np.sqrt(np.einsum("ijk->ij", (A[:, None, :] - B) ** 2))
        """

        # firstly expand a new dimension in the middle by None
        distances = np.sqrt(np.einsum("ijk->ij", (self.data[:, None, :] - self.centroids)**2))
        return distances

    def train(self):
        self.initialize_membership_matrix()

        for _ in range(self.max_iter):
            old_membership_matrix = self.membership_matrix.copy()
            self.find_centroids()
            self.update_membership_matrix()

            # Stopping rule
            if np.linalg.norm(self.membership_matrix - old_membership_matrix) < self.epsilon:
                break

    def _compute_inference_distance(self, inputs):
        distances = np.sqrt(np.einsum("ijk->ij", (inputs[:, None, :] - self.centroids) ** 2))
        return distances

    def get_inference(self, inputs):
        inputs = np.expand_dims(inputs, axis=0) if len(inputs.shape) == 1 else inputs
        dist = self._compute_inference_distance(inputs)
        temp = dist ** float(2 / (self.m - 1))
        denominator_ = temp.reshape((inputs.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
        denominator_ = temp[:, :, np.newaxis] / denominator_
        fuzzy_results = 1 / denominator_.sum(2)
        de_fuzzy_results = fuzzy_results.argmax(axis=-1)
        return de_fuzzy_results