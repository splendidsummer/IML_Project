# Clustering Project Report 

## **Report Requirement**

The report will contain: 
- Details about the implementation of your algorithms, including the decisions made during the implementation and the setup of the different parameters. 
- The evaluation of the algorithms, including tables and/or graphs that show your results with 
comments about them. 
- Justify your results and, in addition, reason each one of the questions defined above in your 
evaluation. Moreover, add any comment or observation that you consider important from your 
results. 
- It is extremely important that you explain how to execute your code. Moreover, call files 
from the relative path to the project, not the global paths of your computer.**


## **Introduction**

Cluster analysis is popular unsupervised learning method which divides data into groups (clusters) that are meaningful, useful, or both. If meaningful groups are the goal, then the clusters should capture the natural structure of the data. There are lots of specific clustering techniques in nowadays. 

### **Hierarchical vs Partitional**

The most commonly discussed distinction among different types of clustering is whether the set of clusters is nested or unnested, or in more traditional terminology, hierarchical or partitional.  
A partitional clustering is simply a division of the set of data objects into non-overlapping subsets (clusters) such that each data object is in exactly one subset. Whereas Hierarchical methods permit clusters to have sub-clusters, then we obtain a hierarchical clustering, which is a set of nested clusters that are organized as a tree. Each node (cluster) in the tree (except for the leaf nodes) is the union of its children (sub-clusters), and the root of the tree is the cluster containing all the objects.    

Basic approaches for generating a hierarchical clustering includes **bottom-up(agglomerative)** and **top-down(divisive)** methodologies. Agglomerative algorithm starts with each example in its own cluster and iteratively combine them to form larger and larger clusters, while divisive algorithm starts with all the examples in a single cluster, and choose the best division by considering all the possible ways to divide the cluster into two. 

### **Exclusive versus Overlapping versus Fuzzy**

In the most general sense, an overlapping or non-exclusive
clustering is used to reflect the fact that an object can simultaneously belong to more than one group (class). For example, in a fuzzy clustering, every object belongs to every cluster with a membership weight that is between 0 (absolutely doesn’t belong) and 1 (absolutely belongs). In other words, clusters are treated as fuzzy sets. (Mathematically, a fuzzy set is one in which an object belongs to any set with a weight that is between 0 and 1. In fuzzy clustering, we often impose the additional constraint that the sum of the weights for each object must equal 1.)    

### **Different methods to group data**  
We could use different methods to group large sets of data into small sets of clusters of similar data, which including follows:

-  Based on connectivity: Hierarchical clustering
- Based on centroids: K-means
- Distribution-based models: Mixture models, Expectation-Maximization
- Density models: DBScan, Optics
- Subspace models: Biclustering
- Group models 
- Graph-based 

### **Road Map to our algorithms**

we use the following two simple, but important techniques
to introduce many of the concepts involved in cluster analysis.   
* **K-means:** This is a prototype-based, partitional clustering technique that attempts to find a user-specified number of clusters (K), which are represented by their centroids.

* **Fuzzy_c_means:** As we have introduced in the earlier chapter, fuzzy clustering is a form of clustering in which each data point can belong to more than one cluster. Similarly, probabilistic clustering techniques compute the probability with which each point belongs to each cluster, and these probabilities must also sum to 1, since a fuzzy or probabilistic clustering does not address true multiclass situations, such as the case of a student employee, where an object belongs to multiple classes.  

* **Comparison in theory:** Fuzzy c-means clustering can be considered a better algorithm compared to the k-Means algorithm. Unlike the k-Means algorithm where the data points exclusively belong to one cluster, in the case of the fuzzy c-means algorithm, the data point can belong to more than one cluster with a likelihood. Fuzzy c-means clustering gives comparatively better results for overlapped data sets.

## Data pre-processing 

####  **Dataset Selection**  

Since k means and fuzzy c means algorithms only works on numerical data values, so we choose our datasets among the datasets of which all feature values is numerical. Based on the requirement, we need to analyse the behaviour of different clustering algorithms in well-known data sets from the UCI repository. But after we check dataset information one by one, we found that most of the datasets which have  tens of features together with a limited number of samples. **For machine learning algorithms, we are actually producing meaningful conclusions based a meaningful sample distribution.** As a rough rule of thumb, your model should train on at least an order of magnitude more examples than trainable parameters. Simple models on large data sets generally beat fancy models on small data sets.

So we select two datasets with considerably larger number of samples compared to others - ***pen-based.arff*** and ***satimage.arff***. 
Pen-base dataset con
|  Dataset Name   | No. of features  | No. of Sample |  
|  ----  | ----  | ----  |
| pen_base  | 16 | 10992 |
| satimage  | 36 | 6435 |

Even the dataset size for these two datasets is closed to each other, but the satimage data set has more than twice the number of features in pen_base dataset. 

### **Data pre-processing pipeline**
Data pre-processing is a predominant step in machine learning to yield highly accurate and insightful results. Greater the quality of data, the greater is the reliability of the produced results. **Incomplete**, **noisy**, and **inconsistent** data are the inherent nature of real-world datasets. Data pre-processing helps in increasing the quality of data by filling in missing incomplete data, smoothing noise, and resolving inconsistencies.

There are many stages involved in data pre-processing: **1)Data Cleaning**, **2)Data Integration**, **3)Data Transformation**, **4)Data Reduction**.   

* **Data cleaning** attempts to impute missing values, smooth out noise, resolve inconsistencies, removing outliers in the data. We have implement **filling NA and dropping NA, dropping duplicates** in the ***"./dataset/data_preprocessing_numerical.py"*** file.   
* **Data integration** integrates data from a multitude of sources into a single data warehouse. We haven't touched this part in our project, but it is critical procedure for merging data sets from different sources.   
* **Data transformations**, such as normalization, may be applied in some cases. The goal of normalization is to change the values of numeric columns in the dataset to a common scale, without distorting differences in the ranges of values or losing information. There are several ways to perform normalization including **standardisation by Z scores**, **normalized into -1 to 1 range by mean value**,  **scaling value between 0 and 1 with max and min values**, **Scaling values by considering the whole feature vector to be of unit length**.      
* **Data reduction** can reduce the data size by dropping out redundant features. Feature selection and feature extraction techniques can be used. We do not cover this topic in work 1, but we will continue further steps for data pre-processing by using data reduction techniques in work 2.

* One extra step we do for data pre-processing is **shuffling data**.  

* We must notice that the **true labels from raw data is either all text or binary format**. So here we utilize LabelEncoder from sklearn library to **transform target labels into value between 0 and n_classes-1**. 

by running the main function in **"./dataset/data_preprocessing_numerical.py"**, we can successfully load and pre-process datasets into 2 separate pickle file, one with purely numerical features and the other one with labels with integer values. 

## **Algorithm implementation and evaluation**

In this project, we are required to implement the following algorithms: 

1. **OPTICS with sklearn library**
1. **K-Means implemented by our own code)**
1. **Fuzzy c means implemented by our own code)**

### **Optics implementation and evaluation**
Ordering points to identify the clustering structure (OPTICS)[1] is an algorithm for finding density-based clusters in spatial data. Its basic idea is similar to DBSCAN,[3] but it addresses one of DBSCAN's major weaknesses: the problem of detecting meaningful clusters in data of varying density. To do so, the points of the database are (linearly) ordered such that spatially closest points become neighbors in the ordering. Additionally, a special distance is stored for each point that represents the density that must be accepted for a cluster so that both points belong to the same cluster. 
OPTICS   


![optics](optics.png)
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Reachability Plot</center>  

Using a ***reachability-plot*** (a special kind of dendrogram), the hierarchical structure of the clusters can be obtained easily. It is a 2D plot, with the ordering of the points as processed by OPTICS on the x-axis and the reachability distance on the y-axis. Since points belonging to a cluster have a low reachability distance to their nearest neighbor, the clusters show up as valleys in the reachability plot. The deeper the valley, the denser the cluster.

for optics algorithms we can call corresponding API as follow:   
```class sklearn.cluster.OPTICS(*, min_samples=5, max_eps=inf, metric='minkowski', p=2, ...)```

There are two parameters, **max_eps** and **eps** which will be used to determine the maximum distance between two samples for one to be considered as in the neighborhood of the other. Based on our understanding, **the epsilon value in OPTICS is solely to limit the runtime complexity when using index structures..** So we choose to use default value which is np.inf to run our model. Another parameter **min_cluster_size** (to determine minimum number of samples in an OPTICS cluster) for which we do not have enough knowledge to tune, so we keep the default values. The only parameter we tune is **min_samples**, here we built a list $[5, 10, 15, 20, 25, 30, 35k, 40, 45, 50]$ to investigate the performance of OPTICS.

* **Performance at different min_samples**
>dfd  
>fds  
>dfd  图图图 
>fds  
>vdf  

### **Kmeans implementation**

The algorithm for kmeans is iterative which groups the dataset into K pre-defined distinct non-overlapping subgroups (clusters) where each data point belongs to a single specific group. The main objective is to make the intra-cluster vector points as equal as possible while also keeping the clusters as far different as possible. It assigns data points to a cluster such that the sum of the squared distance between the data points and the cluster’s centroid (arithmetic mean of all the data points that belong to that cluster) is at the minimum. The less variation we have within clusters, the more homogeneous (similar) the data points are within the same cluster.  

The way our kmeans algorithm works is as follows:

* **Training**

    1. Specify number of clusters K.
    1. Initialize kmeans centroids shuffling the dataset and randomly selecting K amount of values for the centroids.
    1. Keep iterating until there is no change to the centroids. i.e assignment of data points to clusters isn’t changing.
        * Compute the sum of the squared distance between data points and all centroids.
        * Assign each data point to the closest cluster (centroid). 
        * Compute the centroids for the clusters by taking the average of all the data points in each cluster.

* **Inference**
    1. Compute the sum of the squared distance between data points and all centroids.
    2. Assign each data point to the closest cluster (centroid)

Moreover, since there is no right answer in terms of the number of clusters that we should have in any problem, sometimes domain knowledge and intuition may help but usually that is not the case. In this methodology, we decide to evaluate how well the models are performing based on different K clusters by evaluation metrics built by us.  
Our number of clusters setting for kmeans:
1. we use $[2, 3, 4, ..., 15]$ for pen_base dataset to evaluate model performance. 
1. we use $[2, 3, 4, ..., 10]$ for satimage dataset to do the evaluation. 

> **Few things to note here**: Given kmeans iterative nature and the random initialization of centroids at the start of the algorithm, different initializations may lead to different clusters since kmeans algorithm may stuck in a local optimum and may not converge to  global optimum. Therefore, it’s recommended to run the algorithm using different initializations of centroids and pick the results of the run that that yielded the lower sum of squared distance.

## Performance evaluation  

### Introduction to evaluation metrics and method in clustering

#### **Evaluation Metrics**

##### **Purity** 

**Purity** is a simple and transparent evaluation measure, and measures the extent to which a cluster contains objects of a single class. We assign a label to each cluster based on the most frequent class in it. Then the purity becomes the number of correctly matched class and cluster labels divided by the number of total data points.

$$P = \frac{1 \over N }$$
    
    
    \sum_{i=1}^n{x+y}}$$

$$y = x^2 + z^3 \tag{1}$$ 

$h[m,n] = \frac{\sum_{k,l}({(g[k,l]-\overline g)(f[m-k, n-l]-\overline f_{m,n})}}{\left( \sum_{k,l}(g[k,l]-\overline g)^2{\sum_{k,l}(f[m-k,n-l]-\overline f_{m,n})^2} \right)^{0.5}}$

$\sum_\limits{I=1}^{n}x_i-\bar{x}^2}\sum_\limits{I=1}^{n}$

$\sum_1^n$


##### **SSW** 

##### **Rand Index**

##### **Adjusted Rand Index**

##### **F1 score**



#### **Evaluation method**

***Elbow Method***
Elbow method gives us an idea on what a good k number of clusters would be based on the sum of squared distance (SSE) between data points and their assigned clusters’ centroids. We pick k at the spot where SSE starts to flatten out and forming an elbow. We’ll use the geyser dataset and evaluate SSE for different values of k and see where the curve might form an elbow and flatten out.
Silhouette Analysis
Silhouette analysis can be used to determine the degree of separation between clusters. For each sample:
Compute the average distance from all data points in the same cluster (ai).
Compute the average distance from all data points in the closest cluster (bi).
Compute the coefficient:
The coefficient can take values in the interval [-1, 1].
If it is 0 –> the sample is very close to the neighboring clusters.
It it is 1 –> the sample is far away from the neighboring clusters.
It it is -1 –> the sample is assigned to the wrong clusters.
Therefore, we want the coefficients to be as big as possible and close to 1 to have a good clusters. We’ll use here geyser dataset again because its cheaper to run the silhouette analysis and it is actually obvious that there is most likely only two groups of data points.

### **K means evaluation** 

### **Fuzzy c means evaluation result** 








#### 


 [1]Kriegel, Hans-Peter; Kröger, Peer; Sander, Jörg; Zimek, Arthur (May 2011). "Density-based clustering". Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery. 1 (3): 231–240. doi:10.1002/widm.30.
 [2]Mihael Ankerst; Markus M. Breunig; Hans-Peter Kriegel; Jörg Sander (1999). OPTICS: Ordering Points To Identify the Clustering Structure. ACM SIGMOD international conference on Management of data. ACM Press. pp. 49–60. CiteSeerX 10.1.1.129.6542.
 [3]Martin Ester; Hans-Peter Kriegel; Jörg Sander; Xiaowei Xu (1996). Evangelos Simoudis; Jiawei Han; Usama M. Fayyad (eds.). A density-based algorithm for discovering clusters in large spatial databases with noise. Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96). AAAI Press. pp. 226–231.





























