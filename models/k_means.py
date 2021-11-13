import numpy as np


class Kmeans:

    def __init__(self, k_clusters, data_arr, max_iter):
        self.n_samples = data_arr.shape[0]
        self.data = data_arr
        self.k_clusters = k_clusters
        self.centroids = None
        self.gold_standard = data_arr[:,-1]
        self.training_labels = np.full((self.n_samples, 1), -1)
        self.max_iter = max_iter

    def init_centroids(self):
        n_dim = self.data.shape[-1]
        centroids = np.zeros((self.k_clusters, n_dim))
        for j in range(n_dim):
            min_val = self.data[:j].min()
            max_val = self.data[:j].max()
            val_range = (max_val - min_val)
            centroids[:, j] =  min_val + range*np.random.rand(self.k_clusters, 1)
        self.centroids = centroids

    def _get_class_label(self):
        for i in range(self.n_samples):
            sample_distances = np.sqrt(np.sum(np.power(self.centroids - self.data[i][np.newaxis, :]), axis=1))
            curr_label = np.argmin(sample_distances)
            self.training_labels[i] = curr_label

    def update_centroids(self):
        self._get_class_label()
        for cls in range(self.k_clusters):
            mask = self.training_labels == cls
            cls_index = np.zeros(mask)
            curr_cls_arr = self.data[cls_index]
            cls_centroid = curr_cls_arr.mean(axis=0)
            self.centroids[cls, :] = cls_centroid

    def train(self):
        for _ in range(self.max_iter):
            prev_cls_labels = self.training_labels.copy()
            self.update_centroids()
            self._get_class_label()
            if self.training_labels == prev_cls_labels:
                break

    def get_inference(self):
        results = np.full((self.n_samples, 1), -1)
        for i in range(self.n_samples):
            sample_distances = np.sqrt(np.sum(np.power(self.centroids - self.data[i][np.newaxis, :]), axis=1))
            curr_label = np.argmin(sample_distances)
            results[i] = curr_label
        return results







