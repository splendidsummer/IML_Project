import time
import sys
import scipy
import os
import collections

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def plot_scatter_matrix(df, df_name):
    # sns_plot = sns.pairplot(df)
    # fig_name = df_name + '_scatter_plot.png'
    # sns_plot.savefig(fig_name, dpi=400)
    attributes = df.colunns
    scatter_matrix(df[attributes], figsize=(12, 8))
    save_fig("scatter_matrix_plot")


def save_fig(df_name, fig_path='visualization/results',
             tight_layout=True, fig_extension="jpg", resolution=300):

    fig_path = fig_path + df_name + "." + "jpg"
    print("Saving figure", fig_path)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fig_path, format=fig_extension, dpi=resolution)


def looking_for_correlation(df):
    corr_matrix = df.corr()
    for col in df.colunns:
        # corr_matrix[col].sort_values(ascending=False)
        print(corr_matrix[col].sort_values(ascending=False))
    return corr_matrix


def plot_and_fig(X, y, dataset_name, method_name, metric, normalization):
    title = method_name + ' ' + dataset_name + ' ' + metric + ' ' + normalization + ' at different No. of clusters'
    plt.plot(X, y, 'o-', color='b', label=metric)
    plt.xlabel("k_clusters")
    plt.ylabel(metric)
    plt.legend(loc="best")
    plt.title(title)
    # plt.show()
    img_root = 'golden_result_plots'
    image_path = method_name + '_' + dataset_name + '_' + metric + '_' + normalization + '.png'
    image_path = os.path.join(img_root, image_path)
    plt.savefig(image_path)
    plt.close()


if __name__ == '__main__':

    file_root = '../Golden_Evaluation'
    file_names = os.listdir(file_root)
    method_names = ['Kmeans', 'Fuzzycmeans']
    dataset_names = ['pen-based', 'satimage']
    nomalizations = ['MinMax', 'Standard', 'Mean']
    df_labels = ['purity', 'ssw', 'davies_bouldin_score', 'adjusted_rand_score']
    df_results = {}
    paths = []
    keys = []

    for file_name in file_names:
        for method_name in method_names:
            for dataset_name in dataset_names:
                for normalization in nomalizations:
                    splits = file_name.split('_')
                    if splits[0] == method_name and splits[1] == dataset_name and splits[2][:-4] == normalization:
                        df_key = file_name[:-4]
                        df_path = os.path.join(file_root, file_name)
                        paths.append(df_path)
                        keys.append(df_key)
                        df = pd.read_csv(df_path)
                        df_results[df_key] = df

    # get_paths = []
    get_keys = []

    for method_name in method_names:
        for dataset_name in dataset_names:
            for normalization in nomalizations:
                if dataset_name == 'pen-based':
                    k_range_list = range(2, 15, 1)  # k values for pen-base dataset
                elif dataset_name == 'satimage':
                    k_range_list = range(2, 15, 1)  # k value range for satimage
                df_key = method_name + '_' + dataset_name + '_' + normalization
                get_keys.append(df_key)
                df = df_results[df_key]
                df.index = df_labels

                df = df.drop(columns=df.columns[0])
                for idx in df.index:
                    y = df.loc[idx, :]
                    plot_and_fig(k_range_list, y, dataset_name, method_name, idx, normalization)

