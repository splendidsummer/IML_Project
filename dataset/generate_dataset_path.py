# file path
import os
import json

data_path = '../content/drive/Mydrive/imlproject/'
fileids = os.listdir(data_path)
data_json = 'arff_paths_dict.json'
# main_path = 'content/drive/Mydrive/imlproject/'

arff_paths = []
arff_names = []
arff_paths_dict = {}

for i in range(len(fileids)):
    file_path = data_path + fileids[i]
    arff_paths.append(file_path)
    arff_names.append(fileids[i].split('.')[0])

for name, path in zip(arff_names, arff_paths):
    arff_paths_dict[name] = path
arff_paths_dump = json.dumps(arff_paths_dict)

f = open('arff_paths_dict.json', 'w')
f.write(arff_paths_dump)
f.close()

