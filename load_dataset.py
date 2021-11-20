import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(name='', test_rate=0.25):
    """
    load dataset
    :param name: which data to train
    :param test_rate: the test rate of all data
    :return:
    """
    data_root = 'dataset'
    if name == 'pen':
        print('------------------ the cur data set is pen ----------------')
        data_path = os.path.join(data_root, 'pen-based.pkl')
        label_path = os.path.join(data_root, 'pen-based_label.pkl')
    elif name == 'sat':
        print('------------------ the cur data set is sat ----------------')
        data_path = os.path.join(data_root, 'satimage.pkl')
        label_path = os.path.join(data_root, 'satimage_label.pkl')
    data = pickle.load(open(data_path, 'rb'), encoding='utf-8')
    label = pickle.load(open(label_path, 'rb'), encoding='utf-8')

    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=test_rate, stratify=label)
    return data_train, data_test, label_train, label_test
