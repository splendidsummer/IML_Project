import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit,\
    cross_val_score, KFold
from sklearn.utils import shuffle
import json
import pickle


def load_numerical_data(arff_paths_dict, dataset_name):
    data, meta = arff.loadarff(arff_paths_dict[dataset_name])
    df = pd.DataFrame(data)
    return df, meta


def drop_duplicates(df):
    df = df.drop_duplicates()
    print('There are {} duplicate items get dropped')
    return df


def get_null_counts(df):  # count all the null values in df and null values in each column
    df_null_counts = {}
    df_null_all = df.isnull().values().sum()
    df_null_counts['all'] = df_null_all
    for col in df.columns:
        df_null_counts[col] = df[col].isnull().values().sum()

    return df_null_counts


def drop_na(df, how='any', thresh=None):
    if how == 'any':
        df = df.dropna(how=how)
    if how == 'all':  # Passing how='all' will only drop rows that are all NA
        df = df.dropna(how=how)
    if how == 'thresh':
        thresh = int(input("You must input an integer for setting the threshold "))
        df = df.dropna(thresh=thresh)
    return df


def fill_na(df, fill_value=False, inplace=True, method=''):
    """"
    method: filling NA method
            ffill: an interpolation methods available for reindexing can be used with fillna
            mean: filling the missing value by mean value
    """
    if fill_value:
        value = float(input("You need to input a numerical specific value for NA"))
        df = df.fillna(value, inplace=True)
    else:
        if method is None:
            raise ValueError('You must define the fillna method if the fill_value is not given')
        elif method == 'mean':
            for col in list(df.columns):
                mean_value = df[col].mean()
                df[col].fillna(mean_value, inplace=inplace)
        elif method == 'ffill':
            df = df.fillna(method=method, limit=2, inplace=True)

    return df


def normalize_data(df, normalize_method='MinMax'):
    """
    There are three ways in sklearn to do normalization in our data:
    1. scale: Standardisation replaces the values by their Z scores
    2. StandardScale: This distribution will have values between -1 and 1with μ=0
    3. MinMaxScaler: This scaling brings the value between 0  and 1.
    4. Normalizer:
    """

    if normalize_method == 'MinMax':
        normalizer = MinMaxScaler  # This scaling brings the value between 0 and 1
    elif normalize_method == 'Standard':
        normalizer = scale  # Standardisation replaces the values by their Z scores, much more like Gaussian
    elif normalize_method == 'Mean':    # This distribution will have values between 1 and -1 and 1with μ=0
        normalize_method == StandardScaler
    elif normalize_method == 'UnitVector':  # Scaling is done considering the whole feature vector to be of unit length
        normalizer = Normalizer

    normalize_df = df.apply(normalizer)  # Normalize data in each columns according to normalize_method

    return normalize_df


# For illustration only. Sklearn has train_test_split()
def shuffle_data(df):
    df = shuffle_data(df)
    return df


def split_train_test(df, split_ratio, n_splits=5, random_state=0, split_type='random'):
    last_column_name = df.columns[-1]
    df.rename(columns={last_column_name: 'class'})

    if split_type == 'random':
        train_set, test_set = train_test_split(df, test_size=(1-split_ratio), random_state=random_state)

    if split_type == 'stratified':
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(df, df["class"]):
            train_set = df.loc[train_index]
            test_set = df.loc[test_index]

    if split_type == 'cross_validation':
        kf = KFold(n_splits=n_splits, random_state=random_state)

    return train_set, test_set

#
#
# def truncated_value(df, max_value, min_value):
#     truncated_df = None
#     return truncated_df


if __name__ == '__main__':

    dict_path = 'arff_paths_dict.json'
    with open(dict_path, 'r') as f:
        arff_path_dict = json.load(f)

    dataset_names = ['pen-based', 'satimage']
    for i, dataset_name in enumerate(dataset_names):
            data, meta = load_numerical_data(arff_path_dict, dataset_names[i])
            df = pd.DataFrame(data)
            num_nulls = df.isnull().sum()
            print('There are {} null values in {} dataset'.format(num_nulls, dataset_names[i]))
            num1 = len(df)
            df = drop_duplicates(df)
            num2 = len(df)
            print('{} duplicate samples get dropped in {} dataset'.format((num1 - num2), dataset_name))
            df = drop_na(df, how='all')
            num3 = len(df)
            print('{} nan samples get dropped in {} dataset'.format((num2 - num3), dataset_name))
            df = fill_na(df, method='mean')
            df = normalize_data(df, normalize_method='MinMax')  # This scaling brings the value between 0 and 1

            df = shuffle_data(df)
            pickle_name = dataset_name + '.pkl'

            with open(pickle_name, 'wb') as f:
                pickle.dump(pickle_name, df)

