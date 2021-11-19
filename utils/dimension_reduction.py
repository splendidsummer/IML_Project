import sklearn
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
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

    def _get_n_components(self, X):
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

        n_components = self._get_n_components(X)
        self.PCA_coordinats = np.dot(X, n_components)
        return self.PCA_coordinats


##################################################################
# t-SNE
################################################################
class t_SNE:
    def __init__(self, X, no_dims, max_iters, tol):
        self.n_examples, self.init_dim = X.shape
        self.reduced_dim = no_dims
        self.perplexity = None
        self.distance = -1000 * np.ones(())
        self.data = X
        self.max_iters = max_iters
        self.init_  =  np.ones(1, 100)
        self.tol = tol


    def get_distance(data):
        sum_x = np.sum(np.square(data), 1)
        distance = np.add(np.add(-2* np.dot(X, X.T), sum_x), sum_x)
        return distance

    def find_probability(self):
        dist = self.get_distance(self.data)
        pair_probability = np.zeros((self.no_examples, self.no_examples))

    def get_perplexity(self, index=0, beta=1.0):
        distance = self.get_distance(self.data)
        probability = np.exp(-distance * beta)
        probability[index] = 0.
        sum_probability = np.sum(probability)
        perplexity = np.log(sum_probability) + beta + np.sum(distance, )
        self.perplexity

    def dimension_reduction(self):
        distance = self.get_distance(self.data)
        beta = np.zeros((self.n_sample, self.n_sample))
        beta = np.ones((self.n_sample, 1))
        base_perplexity = np.log(self.perplexity)

        for i in range(500):
            if i / 500 == 0:
                print('Calculating the {} out of {} points in total'.format(i, self.n_examples))
            betamin = - np.inf
            betamax = np.inf
            this_probability = self.get_perplexity(distance[i], i, beta[i])
            perplexity_diff = self.perplexity - base_perplexity
            tries = 0
            while np.abs(perplexity_diff) > self.tol and tries < 50:



        perplexity,  curr_probability = getPerplexity()




    def sne_crowding(self):







def sne_crowding():
    npoints = 1000  # 抽取1000个m维球内均匀分布的点
    plt.figure(figsize=(20, 5))
    for i, m in enumerate((2, 3, 5, 8)):
        # 这里模拟m维球中的均匀分布用到了拒绝采样，
        # 即先生成m维立方中的均匀分布，再剔除m维球外部的点
        accepts = []
        while len(accepts) < 1000:
            points = np.random.rand(500, m)
            accepts.extend([d for d in norm(points, axis=1)
                            if d <= 1.0])  # 拒绝采样
        accepts = accepts[:npoints]
        ax = plt.subplot(1, 4, i + 1)
        if i == 0:
            ax.set_ylabel('count')
        if i == 2:
            ax.set_xlabel('distance')
        ax.hist(accepts, bins=np.linspace(0., 1., 50))
        ax.set_title('m=%s' % m)
    plt.savefig("./images/sne_crowding.png")

    x = np.linspace(0, 4, 100)
    ta = 1 / (1 + np.square(x))
    tb = np.sum(ta) - 1
    qa = np.exp(-np.square(x))
    qb = np.sum(qa) - 1


def sne_norm_t_dist_cost():
    plt.figure(figsize=(8, 5))
    plt.plot(qa / qb, c="b", label="normal-dist")
    plt.plot(ta / tb, c="g", label="t-dist")
    plt.plot((0, 20), (0.025, 0.025), 'r--')
    plt.text(10, 0.022, r'$q_{ij}$')
    plt.text(20, 0.026, r'$p_{ij}$')

    plt.plot((0, 55), (0.005, 0.005), 'r--')
    plt.text(36, 0.003, r'$q_{ij}$')
    plt.text(55, 0.007, r'$p_{ij}$')

    plt.title("probability of distance")
    plt.xlabel("distance")
    plt.ylabel("probability")
    plt.legend()
    plt.savefig("./images/sne_norm_t_dist_cost.png")


def cal_pairwise_dist(x):
    '''计算pairwise 距离, x是matrix
    (a-b)^2 = a^w + b^2 - 2*a*b
    '''
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    return dist


def cal_perplexity(dist, idx=0, beta=1.0):
    '''计算perplexity, D是距离向量，
    idx指dist中自己与自己距离的位置，beta是高斯分布参数
    这里的perp仅计算了熵，方便计算
    '''
    prob = np.exp(-dist * beta)
    # 设置自身prob为0
    prob[idx] = 0
    sum_prob = np.sum(prob)
    perp = np.log(sum_prob) + beta * np.sum(dist * prob) / sum_prob
    prob /= sum_prob
    return perp, prob


def seach_prob(x, tol=1e-5, perplexity=30.0):
    '''二分搜索寻找beta,并计算pairwise的prob
    '''

    # 初始化参数
    print("Computing pairwise distances...")
    (n, d) = x.shape
    dist = cal_pairwise_dist(x)
    pair_prob = np.zeros((n, n))
    beta = np.ones((n, 1))
    # 取log，方便后续计算
    base_perp = np.log(perplexity)

    for i in range(n):
        if i % 500 == 0:
            print("Computing pair_prob for point %s of %s ..." % (i, n))

        betamin = -np.inf
        betamax = np.inf
        perp, this_prob = cal_perplexity(dist[i], i, beta[i])

        # 二分搜索,寻找最佳sigma下的prob
        perp_diff = perp - base_perp
        tries = 0
        while np.abs(perp_diff) > tol and tries < 50:
            if perp_diff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # 更新perb,prob值
            perp, this_prob = cal_perplexity(dist[i], i, beta[i])
            perp_diff = perp - base_perp
            tries = tries + 1
        # 记录prob值
        pair_prob[i,] = this_prob
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    return pair_prob


def pca(x, no_dims=50):
    ''' PCA算法
    使用PCA先进行预降维
    '''
    print("Preprocessing the data using PCA...")
    (n, d) = x.shape
    x = x - np.tile(np.mean(x, 0), (n, 1))
    l, M = np.linalg.eig(np.dot(x.T, x))
    y = np.dot(x, M[:, 0:no_dims])
    return y


def tsne(x, no_dims=2, initial_dims=50, perplexity=30.0, max_iter=1000):
    """Runs t-SNE on the dataset in the NxD array x
    to reduce its dimensionality to no_dims dimensions.
    The syntaxis of the function is Y = tsne.tsne(x, no_dims, perplexity),
    where x is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array x should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # 初始化参数和变量
    x = pca(x, initial_dims).real
    (n, d) = x.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    y = np.random.randn(n, no_dims)
    dy = np.zeros((n, no_dims))
    iy = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # 对称化
    P = seach_prob(x, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    # early exaggeration
    P = P * 4
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):
        # Compute pairwise affinities
        sum_y = np.sum(np.square(y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))
        num[range(n), range(n)] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dy[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (y[i, :] - y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dy > 0) != (iy > 0)) + (gains * 0.8) * ((dy > 0) == (iy > 0))
        gains[gains < min_gain] = min_gain
        iy = momentum * iy - eta * (gains * dy)
        y = y + iy
        y = y - np.tile(np.mean(y, 0), (n, 1))
        # Compute current value of cost function
        if (iter + 1) % 100 == 0:
            if iter > 100:
                C = np.sum(P * np.log(P / Q))
            else:
                C = np.sum(P / 4 * np.log(P / 4 / Q))
            print("Iteration ", (iter + 1), ": error is ", C)
        # Stop lying about P-values
        if iter == 100:
            P = P / 4
    print("finished training!")
    return y


##################################################################
# UMAP
################################################################
class UMAP:




# ########################## testing function ####################
if __name__ == '__main__':
#     np.random.seed(100)
#     X = np.random.randn(100, 10)
#     pca = PCA(3)
#     X_pca_coordinates = pca.get_coordinates(X)

    # Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.
    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")
    Y = tsne(X, 2, 50, 20.0)
    from matplotlib import pyplot as plt

    plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    plt.show()



