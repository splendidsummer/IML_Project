import numpy as np
from sklearn.metrics import accuracy_score, adjusted_rand_score,\
    v_measure_score, silhouette_score, davies_bouldin_score, f1_score
# from sklearn.metrics.cluster import contingency_matrix
# sklearn.metrics.cluster.contingency_matrix(labels_true, labels_pred, *, eps=None, sparse=False)


def evaluate_on_purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def evaluate_adjusted_rand_score(self):
    rand_score = adjusted_rand_score(self.gold_standard, self.training_labels)
    return rand_score


def evaluate_silhouette_score(self, data_arr, train_labels):
    silhouette_coefficient = silhouette_score(data_arr, train_labels)
    return silhouette_coefficient


def evaluate_on_ssw(data_arr, k_clusters, centroids, training_labels, n_samples):
    ssw = 0.0
    for cls in range(k_clusters):
        centroid = centroids[cls]
        cls_index = np.nonzeros(training_labels == cls)
        curr_cls_arr = data_arr[cls_index]
        ssw += np.sum(np.square(curr_cls_arr - centroid[np.newaxis, :]))
    ssw = ssw / n_samples

    return ssw


def evaluate_on_davies_bouldin_score(data_arr, train_labels):
    db_score = davies_bouldin_score(data_arr, train_labels)
    return db_score
