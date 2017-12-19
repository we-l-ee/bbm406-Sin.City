import librosa as lb
import librosa.display
import numpy as np
import os
import sys

root_dir = sys.argv[1]
features = []
labels = list()
part_in_seconds = 10
#max_part = 5
# encoded_labels = {'akdeniz': 0, 'doguanadolu': 1, 'ege': 2, 'guneydoguanadolu': 3, 'icanadolu': 4, 'karadeniz': 5, 'trakya': 6}
encoded_labels = dict()
label_counter = 0
train_subsetn = 10


def read_train_instructions(ftrain="train.txt"):
    with open(ftrain) as f:
        x, y = [], []
        arr = f.read().splitlines()
        for i in arr:
            d = i.split("\t")
            x.append(d[1])
            y.append(d[0])
    return np.array(x), np.array(y)


def trainset(xx,yy):
    set = dict()
    for x, y in zip(xx, yy):
        try:
            set[y].append(x)
        except KeyError:
            set[y] = [x]
    return set


def train_subsetn_random(set, train_subsetn=10):
    x, y = [], []

    for l in set:
        xx = np.array(set[l])
        indices = list(range(len(xx)))
        xx = xx[np.random.shuffle(indices)]
        xx = xx[:train_subsetn]
        yy = np.empty(train_subsetn)
        yy.fill(l)
        x.extend(xx.tolist())
        y.extend(yy.tolist())
    return np.array(x), np.array(y)


def extract():
    label_counter = 0
    for dir_name, subdir_list, file_list in os.walk(root_dir):
        for f_name in file_list:
            file_path = os.path.join(dir_name, f_name)
            print(file_path)
            label = file_path.split('\\')[1]

            try:
                y, sr = lb.load(file_path)
                duration = lb.core.get_duration(y=y, sr=sr)
                parts = int(duration/part_in_seconds)
            except AssertionError:
                continue

            mfc = lb.feature.melspectrogram(y=y, sr=sr).T
            log_S = librosa.logamplitude(mfc, ref_power=np.max)

            size = len(log_S)/parts

            for i in range(0, len(log_S), int(size)):
                features.append(log_S[i:i+int(size)])
                if label in encoded_labels:
                    labels.append(encoded_labels[label])
                else:
                    labels.append(label_counter)
                    encoded_labels[label] = label_counter
                    label_counter += 1


x, y = read_train_instructions()
train_set = trainset(x, y)
xx, yy = train_subsetn_random(train_set)

