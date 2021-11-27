import os
import time
import random

import pandas as pd

from models.k_means import *
from utils.evaluate_utils import *
from utils.utils_function import *
from sklearn.metrics import adjusted_rand_score, v_measure_score,\
    silhouette_score, davies_bouldin_score, f1_score
import json
from sklearn.model_selection import KFold, StratifiedKFold
from collections import defaultdict


if __name__ == '__main__':

    data_index = 0  # Change here to choose different dataset
    data_name_list = ['pen-based', 'satimage']
    dataset_name = data_name_list[data_index]
    average_times = 3

    metrics_list = ['purity', 'ssw', 'davies_bouldin_score', 'adjusted_rand_score', 'silhouette_score']
    # data_train, data_test, label_train, label_test = load_split_data(name=dataset_name, test_rate=0.25)

    if data_index == 0:
        k_range_list = range(2, 15, 1)  # k va
        # lue range for pen_based dataset
    elif data_index == 1:
        k_range_list = range(2, 11, 1)  # k value range for satimage dataset

    result_arr = np.zeros((len(k_range_list), len(metrics_list)))
    df_results = pd.DataFrame(result_arr, columns=metrics_list, index=k_range_list)

    # train and save the results
    n_splits = 5
    metrics_result_list = []
    score = 0
    result = defaultdict(dict)
    random_seed = np.arange(len(metrics_list)*len(k_range_list))*100 + 1

    for i, metric in enumerate(metrics_list):
        for j, k_clusters in enumerate(k_range_list):
            random_idx = i*len(k_range_list) + j
            random_state = random_seed[random_idx]

            print('the cur k value for clustering is', k_clusters)
            data, labels = load_data(dataset_name)
            kfold = KFold(n_splits=n_splits)
            kfold_idx = 1
            results = []
            start_time = time.time()
            for train_idx, test_idx in kfold.split(data, labels):
                train_data, train_labels = data[train_idx], labels[train_idx]
                test_data, test_labels = data[test_idx], labels[test_idx]
                print('------------- start training kfold {} '.format(kfold_idx))
                cluster = Kmeans(k_clusters=k_clusters, data_arr=train_data, max_iter=1000, random_state=random_state)
                cluster.init_centroids()
                cluster.train()
                y_pred = cluster.get_inference(test_data)

                if metric == 'davies_bouldin_score':
                    score += davies_bouldin_score(test_data, test_labels)/n_splits

                if metric == 'adjusted_rand_score':
                    score = adjusted_rand_score(y_pred, test_labels)/n_splits

                if metric == 'purity':
                    score += evaluate_on_purity_score(y_pred, test_labels)/n_splits

                elif metric == 'ssw':
                    tempv = evaluate_on_ssw(data_arr=test_data, k_clusters=k_clusters,
                                            centroids=cluster.centroids, training_labels=test_labels,
                                            n_samples=test_labels.shape[0])
                    score += tempv/n_splits

                kfold_idx += 1

            results.append(score)

        df_results.loc[metric] = results
        csv_path = 'Kmeans {} dataset evaluation result.csv'.format(dataset_name)
        df_results.to_csv('data_df.csv')







