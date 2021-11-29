import numpy as np
import math
def loadDataSet(fileName):
    """
    函数说明:加载数据
    Parameters:
        fileName - 文件名
    Returns:
        dataMat - 数据矩阵
    """
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # 转化为float类型
        dataMat.append(fltLine)
    return np.array(dataMat)


def distEclud(vecA, vecB):
    """
    函数说明：欧拉距离
    parameters：
         vecA,vecB:两个数据点的特征向量
    returns：
         欧式距离
    """
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCent(dataSet, k):
    """
    函数说明：
    :param dataSet: 数据矩阵
    :param k: 最终分类的个数
    :return: centroids：一个包含k个随机质心的集合
    """
    # n为特征值个数
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        # minJ为特征值最小值，rangeJ为特征值取值范围
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    """
    函数说明：kMeans算法的实现
    :param dataSet:数据矩阵
    :param k:待分类的类别数
    :param distMeas:度量距离的公式
    :param createCent:随机创建的初始点
    :return:
        centroids:聚类算法
        clusterAssment：聚类结果
    """
    m = np.shape(dataSet)[0]
    # clusterAssment用来存储聚类中心
    clusterAssment = np.mat(np.zeros((m,2)))
    centroids = createCent(dataSet, k)
    # clusterChanged用来判断算法是否收敛
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # 遍历所有数据，将其分入最近的聚类中心中
        for i in range(m):
            # minIndex记录该点属于的类别
            minDist = float(math.inf)

            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
    #         # 判断是否收敛
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        # 重新计算聚类中心
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


dataMat = loadDataSet("E:\\testSet.txt")
centroids, clusterAssment = kMeans(dataMat, 3)
print(centroids,clusterAssment)