from models.fuzzy_c_means import *
import torch
from sklearn.model_selection import KFold, RepeatedKFold, LeaveOneOut

random_state = 2**3 + 1

inputs1 = np.random.randn(100, 10)
inputs2 = np.random.randn(100, 10) + 10.0
inputs3 = np.random.randn(100, 10) + 5.0
inputs = np.vstack((inputs1, inputs2))
inputs = np.vstack((inputs, inputs3))

test_data1 = np.random.randn(10, 10)
test_data2 = np.random.randn(10, 10) + 5.
test_data3 = np.random.randn(10, 10) + 10.
test_data = np.vstack((test_data1, test_data2))
test_data = np.vstack((test_data, test_data3))

labels = np.ones((300, 1))

kfold = KFold(n_splits=5)

for train_idx, test_idx in kfold.split(inputs, labels):
    train_data = inputs[train_idx]
    test_data = inputs[test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    print(train_data.shape)
    print(test_data.shape)

#
# cluster = fuzzyCMeans(3, inputs, 33, max_iter=100, epsilon=1e-5)
#
# cluster.train()
# inferences = cluster.get_inference(test_data)




