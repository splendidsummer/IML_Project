"""
This file here is for clustering performance evaluation,
which may include the following methods:

– Adjusted Rand index
– Mutual information based scores
– Homogeneity, completeness and V-measure
– Fowlkes-Mallows scores
– Silhouette Coefficient
– Calinski-Harabaz Index
– Davies-Bouldin Index
– Contingency Matrix

"""

# SSW for elbow searching -- internal

from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score  # external evaluation
from sklearn.metrics import f1_score, davies_bouldin_score

def sum_in_cluster(centroid, samples):


metrics.silhouette_score()













