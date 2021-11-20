import os
import time
import random
from models.fuzzy_c_means import *
from load_dataset import load_data
from utils.evaluate_utils import *

if __name__ == '__main__':

    random_states = [66, 88, 168]
    for random_state in random_states:
        data_index = 0
        data_name_list = ['pen', 'sat']
        dataset_name = data_name_list[data_index]

        metrics_list = ['purity', 'ssw', 'davies_bouldin_score', ]
        metrics_index = 2
        metric = metrics_list[metrics_index]

        data_train, data_test, label_train, label_test = load_data(name=dataset_name, test_rate=0.25)

        if data_index == 0:
            range_list = range(2, 15, 1)
        elif data_index == 1:
            range_list = range(2, 10, 1)

        # train and save the results
        metrics_result_list = []
        k_list = []
    # for k_clusters in range_list:
        for k_clusters in range_list:
            print('the cur k value for clustering is ', k_clusters)
            random_state = random.randint(1, 1000)
            cluster = fuzzyCMeans(k_clusters, data_train, random_state, max_iter=100, m=2, epsilon=1e-5)
            cluster.initialize_membership_matrix()

            print('------------ start training------------')
            start_time = time.time()
            cluster.train()
            print('-------the training cost:{}s----------------'.format(time.time()-start_time))

            y_pred = cluster.get_inference(data_test)
            if metric == 'purity':
                purity = evaluate_on_f1_score_score(y_pred, label_test)
                metrics_result_list.append(purity)
                k_list.append(k_clusters)

            elif metric == 'ssw':
                ssw = evaluate_on_ssw(data_arr=data_test, k_clusters=k_clusters,
                                      centroids=cluster.centroids, training_labels=label_test,
                                      n_samples=label_test.shape[0])

                metrics_result_list.append(ssw)
                k_list.append(k_clusters)

        print(metrics_result_list, k_list)
        # print(purity)