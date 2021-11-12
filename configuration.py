import json
import argparse


splice_df_path = 'dataset/splice_df.csv'
meta_attribute_dict_path = 'dataset/meta_attribute_dict.pkl'
df_attribute_dict_path = 'dataset/df_attribute_dict.pkl'
arff_paths = 'dataset/arff_paths_dict.json'
encoded_unique_dict_path = 'dataset/encoded_unique_values.pkl'
splice_array_path = 'dataset/splice_array.pkl'

# import file paths for all the training datasets here
file_path = 'dataset/arff_paths_dict.json'
with open(file_path, 'r') as path_dict:
    arff_paths_dict = json.load(path_dict)

parser = argparse.ArgumentParser(description='Training and evaluation parameters setting.')

##########################################################
# general configuration
parser.add_argument('n_clusters', type=int, help='number of clusters for k means or k modes or k prototypes')
parser.add_argument('n_components', type=int, help= 'number of componets in PCA')

parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')



print(args.accumulate(args.integers))



n_clusters = 3
max_iteration = 100
init = ''
n_init = 10
random_seeds = [i*1000 - 2 for i in list(range(n_init))]

best_centroids_path = 'best_centroids.pkl'








