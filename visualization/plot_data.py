import time
import sys
import collections
from io import StringIO
import scipy
import os
import random
import re

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

import matplotlib
matplotlib.use('Agg')
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


def plot_and_fig(X, y, dataset_name, data_root, metric):
    title = metric + 'at different number of clusters (k_clusters)'
    plt.plot(X, y, 'o-', color='b', label=metric)
    plt.xlabel("k_clusters")
    plt.ylabel(metric)
    plt.legend(loc="best")
    plt.title(title)
    plt.show()
    # data_root = ['./results/kmeans', './results/fuzzycmean']
    image_path = data_root + dataset_name + '_' + metric + '.jpg'
    plt.savefig(image_path)
