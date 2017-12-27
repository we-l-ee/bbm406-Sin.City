import numpy as np
import os
import sys

root_dir = sys.argv[1]

dataset = []
label_counter = 0
labels = dict()
for dir_name, subdir_list, file_list in os.walk(root_dir):
    if len(subdir_list) > 0:
        continue
    for f_name in file_list:
        cl = dir_name.split("\\")[-1]
        file_path = os.path.join(dir_name, f_name)
        print(file_path ,dir_name)
        try:
            label = labels[cl]
        except KeyError:
            labels[cl] = label_counter
            label = labels[cl]
            label_counter += 1
        dataset.append([label, file_path])

f = open(sys.argv[2], 'w')
for d in dataset[:-1]:
    f.write(str(d[0])+"\t"+d[1]+"\n")
f.write(str(dataset[-1][0]) + "\t" + dataset[-1][1])

if not os.path.exists("model"):
    os.mkdir('model')

np.save("model/labels.npy", [labels])
f.close()
