import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit,\
    cross_val_score, KFold



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
        df = df.fillna(value, inplace=inplace)
    else:
        if method is None:
            raise ValueError('You must define the fillna method if the fill_value is not given')
        elif method == 'mean':
            df_col_mean = df.mean()
            df = df.fillna(df_col_mean)
        elif method == 'ffill':
            df = df.fillna(method=method, limit=2)

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
        normalizer = MinMaxScaler
    elif normalize_method == 'Standard':
        normalizer = scale  # Standardisation replaces the values by their Z scores
    elif normalize_method == 'Mean':    # This distribution will have values between -1 and 1with μ=0
        normalize_method == StandardScaler
    elif normalize_method == 'UnitVector':
        normalizer = Normalizer

    normalize_df = df.apply(normalizer)  # Normalize data in each columns according to normalize_method

    return normalize_df


# For illustration only. Sklearn has train_test_split()
def shuffle_data(data_arr, test_ratio):
    shuffled_indices = np.random.permutation(len(data_arr))
    data_arr = data_arr[shuffled_indices]
    return data_arr


def split_train_test(df, split_ratio, n_splits=5, random_state=0, split_type='random'):
    last_column_name = df.columns[-1]
    df.rename(columns={last_column_name: 'class'})

    if split_type == 'random':
        if
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
