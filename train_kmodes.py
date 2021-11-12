
from utils.utils_function import *
from configuration import *
from models import kmodes
import pickle
from tqdm import tqdm


if __name__ == '__main__':

    dissim = matching_dissim

    results = []

    with open(splice_array_path, 'rb') as f:
        X = pickle.load(splice_array_path)

    for single_seed in random_seeds:
        result = kmodes.k_modes(X, n_clusters, max_iteration, dissim, single_seed, random=False)
        results.append(result)

    all_centroids, all_labels, all_costs, all_n_iters, all_epoch_costs = zip(*results)

    best = np.argmin(all_costs)

    if n_init > 1:
        print("Best run was number {}".format(best + 1))

    with open(best_centroids_path) as f:
        pickle.dump(f)














