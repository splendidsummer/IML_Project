import pandas as pd
import numpy as np
from scipy.io import arff
import os


def load_numerical_data(dataset_path):
    data, meta = arff.loadarff(dataset_path)
    df = pd.DataFrame(data)
    return df, meta


curr_dir = os.listdir('raw_data')
print(curr_dir)
df_list = []

for i in curr_dir:
    print('dataset name is {}'.format(i));
    dataset_path = 'raw_data/' + i
    df, _ = load_numerical_data(dataset_path)
    df_list.append(df)

for i, df in enumerate(df_list):
    print('dataset name is {}'.format(curr_dir[i]));
    print(df.info())

