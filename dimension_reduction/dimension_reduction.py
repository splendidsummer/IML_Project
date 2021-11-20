import os
import pickle
import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
plt.style.use('ggplot')


########################################################
# PCA
########################################################
class PCA:
    def __init__(self, n_dimension):
        self.n_dim = n_dimension
        self.PCA_coordinats = None

    def get_n_components(self, X):
        n_components = np.zeros((self.n_dim, X.shape[1]))
        transposed_X = X.T
        # assert np.allclose(X.mean(axis=0), np.zeros(m))
        covariance_matrix = np.dot((transposed_X - transposed_X.mean(axis=1, keepdims=True)), (X - X.mean(axis=0, keepdims=True)))
        eigen_vals, eigen_vecs = np.linalg.eig(covariance_matrix)  # eigenvetor in cols
        orderd_indices = np.flip(np.argsort(eigen_vals))
        sorted_eigen_vals = eigen_vals[orderd_indices]
        sorted_eigen_vecs = eigen_vecs[orderd_indices]
        if eigen_vecs.shape[1] < self.n_dim:
            print('Error: number of dimensions can not exceed the number of eigenvectors!')
            return
        print(sorted_eigen_vals)
        n_components = sorted_eigen_vecs[:, :self.n_dim]
        return n_components

    def get_coordinates(self, X):
        n_components = self.get_n_components(X)
        self.PCA_coordinats = np.dot(X, n_components)
        return self.PCA_coordinats


def load_data(n_dimension, name=''):
    data_root = '../dataset'

    if name == 'pen':
        print('------------------ the cur data set is pen ----------------')
        data_path = 'pen-based.pkl'
        data_path = os.path.join(data_root, data_path)

    elif name == 'sat':
        print('------------------ the cur data set is sat ----------------')
        data_path = 'satimage.pkl'
        data_path = os.path.join(data_root, data_path)

    data = pickle.load(open(data_path, 'rb'), encoding='utf-8')
    return data


def save_data(data, n_dimension, name=''):
    data_root = 'PCA_dataset'
    if name == 'pen':
        print('------------------ the cur data set is pen ----------------')
        n_components_path = 'pen_based_{}_components.pkl'.format(n_dimension)
        data_path = os.path.join(data_root, n_components_path)
    elif name == 'sat':
        print('------------------ the cur data set is sat ----------------')
        n_components_path = 'satimage_{}_components.pkl'.format(n_dimension)
        data_path = os.path.join(data_root, n_components_path)
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    data_name = ['pen', 'sat']
    n_components = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for name in data_name:
        data = load_data(data_name, name=name)
        for n_component in n_components:
            pca = PCA(n_component)
            processed_data = pca.get_n_components(data)
            save_data(processed_data, n_component, name=name)



