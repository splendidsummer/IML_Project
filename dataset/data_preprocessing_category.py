import time
import sys
from io import StringIO
import scipy
from scipy.io import arff
import os
import random
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def byte_to_string(b):
    return str(b, encoding='utf-8')


def load_category_data(arff_paths_dict, dataset_name):
    data, meta = arff.loadarff(arff_paths_dict[dataset_name])
    df = pd.DataFrame(data)
    df = df.\
        map(byte_to_string)
    return df, meta


def process_nominal(meta_item, dataset_name):
    if dataset_name == 'splice':
        nominal_describe = meta_item
    nominal_describe_lst = nominal_describe.split()
    nominal_name = nominal_describe_lst[0][:-2]
    value_regex = re.compile(r'\((.*)\)')
    values_text = value_regex.search(nominal_describe)
    values_text = values_text.group()[2:-2]
    values_text = re.sub(r'\W+', ' ', values_text)
    values = np.array(list(set(values_text.split())))
    return nominal_name, values


def get_meta_attributes(meta, dataset_name):
    meta_lst = meta.__str__().split('\n')[1:]
    meta_lst = meta_lst[:-1]
    meta_attr_value = {}
    if dataset_name is 'splice':
        meta_lst = meta_lst[:-1]
        for meta_item in meta_lst:
            attr_name, attr_values = process_nominal(meta_item, dataset_name)
            meta_attr_value[attr_name] = attr_values
    return meta_attr_value


def get_df_attributes(df):
    attributes = df.columns
    unique_col_values = {}

    for col_name in df.columns:
        col = df[col_name]
        unique_values = col.unique()
        unique_col_values[col_name] = unique_values
    return unique_col_values


#
