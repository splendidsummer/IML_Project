from sklearn.cluster import KMeans
import numpy as np
from utils.utils_function import *

from sklearn.metrics import adjusted_rand_score, v_measure_score,\
    silhouette_score, davies_bouldin_score, f1_score

from utils.evaluate_utils import *

data_index = 1  # Change here to choose different dataset
data_name_list = ['pen-based', 'satimage']
dataset_name = data_name_list[data_index]

normalize_method = 'Mean'

data, labels = load_normal_data(dataset_name, normalize_method)
k_range_list = range(2, 15, 1)
average_time = 3

metrics_list = ['purity', 'ssw', 'davies_bouldin_score', 'adjusted_rand_score']

result_arr = np.zeros((len(metrics_list), len(k_range_list)))
df_results = pd.DataFrame(result_arr, columns=k_range_list, index=metrics_list)

for k_clusters in k_range_list:

    db_score = 0.0
    ar_score = 0.0
    purity_score = 0.0
    ssw_score = 0.0

    for _ in range(average_time):
        kmeans = KMeans(n_clusters=k_clusters, max_iter=1000).fit(data)

        for metric in metrics_list:
            if metric == 'davies_bouldin_score':
                db_score += davies_bouldin_score(data, kmeans.labels_)

            if metric == 'adjusted_rand_score':
                ar_score += adjusted_rand_score(labels, kmeans.labels_)

            if metric == 'purity':
                purity_score += purity_score_func(labels, kmeans.labels_)
                # print(purity_score)

            elif metric == 'ssw':
                ssw_score += evaluate_on_ssw(data_arr=data, k_clusters=k_clusters,
                                             centroids=kmeans.cluster_centers_, training_labels=kmeans.labels_,
                                             n_samples=kmeans.labels_.shape[0])

    db_score = db_score / average_time
    ar_score = ar_score / average_time
    purity_score = purity_score / average_time
    ssw_score = ssw_score / average_time

    results = {'ssw': ssw_score, 'davies_bouldin_score': db_score, 'purity': purity_score, 'adjusted_rand_score': ar_score }

    for metric in metrics_list:
        df_results.loc[metric, k_clusters] = results[metric]

    df_results.to_csv('Kmeans_{}_dataset_{}_normalized_result.csv'.format(dataset_name, normalize_method))
