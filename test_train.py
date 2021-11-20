from models.k_means import *

random_state = 2**3 + 1

inputs1 = np.random.randn(100, 10)
inputs2 = np.random.randn(100, 10) + 10.0
inputs3 = np.random.randn(100, 10) + 5.0

test_data1 = np.random.randn(10, 10)
test_data2 = np.random.randn(10, 10) + 5.
test_data = np.vstack((test_data1, test_data2))


inputs = np.vstack((inputs1, inputs2))
inputs = np.vstack((inputs, inputs3))

cluster = Kmeans(3, inputs, 100, random_state)

cluster.init_centroids()
cluster.train()
inferences = cluster.get_inference(test_data)




