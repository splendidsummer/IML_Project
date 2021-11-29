import os
import time
import pandas as pd
from models.k_means import *
from utils.evaluate_utils import *
from utils.utils_function import *
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
from sklearn.model_selection import KFold, StratifiedKFold


if __name__ == '__main__':
    data_index = 0  # Change here to choose different dataset
    data_name_list = ['pen-based', 'satimage']
    dataset_name = data_name_list[data_index]

    metrics_list = ['purity', 'ssw', 'davies_bouldin_score', 'adjusted_rand_score']

    if data_index == 0:
        k_range_list = range(2, 15, 1)  # k values
    elif data_index == 1:
        k_range_list = range(2, 11, 1)  # k value range for satimage dataset

    result_arr = np.zeros((len(metrics_list), len(k_range_list)))
    df_results = pd.DataFrame(result_arr, columns=k_range_list, index=metrics_list)

    n_training = 3
    random_seed = np.arange(n_training)*1000 + 1

    normalize_method = 'MinMax'
    data, labels = load_normal_data(dataset_name, normalize_method)

    for i, k_clusters in enumerate(k_range_list):

        db_score, ar_score, purity_score,  ssw_score = 0.0, 0.0, 0.0, 0.0
        print('the cur k value for clustering is', k_clusters)
        start_time = time.time()

        for j in range(n_training):
            random_state = random_seed[j]
            cluster = Kmeans(k_clusters=k_clusters, data_arr=data, max_iter=1000, random_state=random_state)
            cluster.init_centroids()
            cluster.train()
            y_pred = cluster.get_inference(data)

            for i, metric in enumerate(metrics_list):
                if metric == 'davies_bouldin_score':
                    db_score += davies_bouldin_score(data, y_pred)
                if metric == 'adjusted_rand_score':
                    ar_score += adjusted_rand_score(labels, y_pred)
                if metric == 'purity':
                    purity_score += purity_score_func(labels, y_pred)

                elif metric == 'ssw':
                    ssw_score += evaluate_on_ssw(data_arr=data, k_clusters=k_clusters,
                                                 centroids=cluster.centroids, training_labels=y_pred,
                                                 n_samples=labels.shape[0])


        end_time = time.time()
        time_spent_kfolds = (end_time - start_time)
        print('{} clusters {} dataset: training and evaluation time is {:.4f}'.format(k_clusters, dataset_name, time_spent_kfolds))

        db_score = db_score / n_training
        ar_score = ar_score / n_training
        purity_score = purity_score / n_training
        ssw_score = ssw_score / n_training

        results = {'ssw': ssw_score, 'davies_bouldin_score': db_score, 'purity': purity_score,
                   'adjusted_rand_score': ar_score}
        for metric in metrics_list:
            df_results.loc[metric, k_clusters] = results[metric]
        df_results.to_csv('Kmeans_{}_dataset_{}_normalized_result.csv'.format(dataset_name, normalize_method))








