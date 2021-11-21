import os
import time
import random
from models.fuzzy_c_means import *
from load_dataset import load_data
from utils.evaluate_utils import *
from sklearn.metrics import adjusted_rand_score, v_measure_score,\
    silhouette_score, davies_bouldin_score, f1_score
import json
from collections import defaultdict


if __name__ == '__main__':
    data_index = 0
    data_name_list = ['pen-based', 'satimage']
    dataset_name = data_name_list[data_index]
    average_times = 3

    metrics_list = ['purity', 'ssw', 'davies_bouldin_score', 'adjusted_rand_score', 'silhouette_score']

    data_train, data_test, label_train, label_test = load_data(name=dataset_name, test_rate=0.25)

    if data_index == 0:
        range_list = range(2, 15, 1)
    elif data_index == 1:
        range_list = range(2, 11, 1)

    # train and save the results
    metrics_result_list = []
    score = 0
    result = defaultdict(dict)
    # for k_clusters in range_list:
    for metric in metrics_list:
        for k_clusters in range_list:
            print('the cur k value for clustering is ', k_clusters)
            for _ in range(average_times):
                random_state = random.randint(1, 1000)
                cluster = fuzzyCMeans(k_clusters=k_clusters, data_arr=data_train, max_iter=100, random_state=random_state)
                cluster.find_centroids()

                print('------------- start training------------')
                start_time = time.time()
                cluster.train()

                print('-------the training cost:{}s----------------'.format(round(time.time()-start_time)))
                y_pred = cluster.get_inference(data_test)

                if metric == 'davies_bouldin_score':
                    score += davies_bouldin_score(data_test, label_test)/average_times

                if metric == 'adjusted_rand_score':
                    score = adjusted_rand_score(y_pred, label_test)/average_times

                if metric == 'purity':
                    score += evaluate_on_purity_score(y_pred, label_test)/average_times

                elif metric == 'ssw':
                    tempv = evaluate_on_ssw(data_arr=data_test, k_clusters=k_clusters,
                                            centroids=cluster.centroids, training_labels=label_test,
                                            n_samples=label_test.shape[0])
                    score += tempv/average_times

            metrics_result_list.append(score)

            result[metric][k_clusters] = metrics_result_list

    print(result)

    json_path = dataset_name + '_fuzzycmeans_evaluation_results.json'

    with open(json_path, "w") as f:
        json.dump(result, f)





