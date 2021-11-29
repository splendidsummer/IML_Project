import numpy as np


class Kmeans:

    def __init__(self, k_clusters, data_arr, max_iter, random_state):
        self.n_samples = data_arr.shape[0]
        self.data = data_arr
        self.k_clusters = k_clusters
        self.centroids = None
        self.training_labels = np.full(self.n_samples, -1)
        self.max_iter = max_iter
        self.rng = np.random.RandomState(random_state)

    def init_centroids(self):
        n_dim = self.data.shape[-1]
        centroids = np.zeros((self.k_clusters, n_dim))
        for j in range(n_dim):
            min_val = self.data[:, j].min()
            max_val = self.data[:, j].max()
            val_range = (max_val - min_val)
            centroids[:, j] = min_val + val_range * self.rng.rand(self.k_clusters)
        self.centroids = centroids

    def _get_class_label(self):
        for i in range(self.n_samples):
            sample = self.data[i]
            differences = self.centroids - sample[np.newaxis, :]
            sample_distances = np.sqrt(np.power(differences, 2).sum(axis=1))
            curr_label = np.argmin(sample_distances)
            self.training_labels[i] = curr_label

    def update_centroids(self):
        for cls in range(self.k_clusters):
            mask = (self.training_labels == cls)
            cls_index = np.nonzero(mask)
            curr_cls_arr = self.data[cls_index]
            cls_centroid = curr_cls_arr.mean(axis=0)
            self.centroids[cls, :] = cls_centroid

    def train(self):
        for i in range(self.max_iter):
            # print(i)
            prev_cls_labels = self.training_labels.copy()
            self._get_class_label()
            self.update_centroids()
            if (self.training_labels == prev_cls_labels).all():
                break

    def get_inference(self, new_data):
        results = np.full(new_data.shape[0], -1)
        for i in range(new_data.shape[0]):
            sample_distances = np.sqrt(np.sum(np.power(self.centroids - new_data[i][np.newaxis, :], 2), axis=1))
            curr_label = np.argmin(sample_distances)
            results[i] = curr_label
        return results








