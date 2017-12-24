import librosa as lb
import numpy as np
import sys
import time
import soundfile as sf
import subprocess

# root_dir = sys.argv[1]
part_in_seconds = 10
# max_part = 5
# encoded_labels = {'akdeniz': 0, 'doguanadolu': 1, 'ege': 2, 'guneydoguanadolu': 3, 'icanadolu': 4, 'karadeniz': 5,
# 'trakya': 6}
encoded_labels = dict()
label_counter = 0
train_subsetn = 10


test_percent = 25
train_percent = 75


def read_instructions(dataset="dataset.txt"):

    with open(dataset) as f:
        np.random.seed(int(time.time()))
        data = f.readlines()
        np.random.shuffle(data)
        train_set = data[:int(len(data) * train_percent / 100)]
        test_set = data[len(train_set):len(train_set)+int(len(data) * test_percent / 100)]

        x_train, y_train = split_set(train_set)
        x_test, y_test = split_set(test_set)

    return x_train, y_train, x_test, y_test


def split_set(set):
    x, y = [], []
    for i in set:
        d = i.split("\t")
        x.append(d[1])
        y.append(int(d[0]))
    return np.array(x), np.array(y)


def data_set(xx, yy):
    set = dict()
    for x, y in zip(xx, yy):
        try:
            set[y].append(x)
        except KeyError:
            set[y] = [x]
    return set


def subsetn_random(set, train_subsetn=10):
    x, y = [], []
    np.random.seed(int(time.time()))
    for l in set:
        xx = np.array(set[l], dtype=object)
        np.random.shuffle(xx)
        xx = xx[:train_subsetn]
        yy = np.empty(train_subsetn)
        yy.fill(l)
        x.extend(xx)
        y.extend(yy)
    return np.array(x), np.array(y)


def extract_train(_paths, _labels):
    labels = []
    features = []
    for p, l in zip(_paths, _labels):
        print(p)

        try:
            if p[-1] == '\n':
                y, sr = lb.load(p[:-1], res_type='kaiser_fast')
            else:
                y, sr = lb.load(p, res_type='kaiser_fast')
            duration = y.shape[0] / sr
            parts = int(duration / part_in_seconds)
            part_len = sr*part_in_seconds
        except AssertionError:
            continue
        except RuntimeError:
            continue

        for i in range(0, parts*part_len, part_len):

            mfc = lb.feature.melspectrogram(y=y[i:i+part_len], sr=sr).T
            log_S = lb.logamplitude(mfc, ref_power=np.max)
            features.append(log_S)
            labels.append(l)

        # print(duration)
        # print(len(log_S))

        # size = int(len(log_S) / parts)
        # #print('size:',size,'spect len:',len(log_S))
        # for i in range(0, len(log_S) - size, size):
        #     frame = log_S[i:i + size].copy()
        #     if row == -1:
        #         row = frame.shape[0]
        #     else:
        #         if row < frame.shape[0]:
        #             frame = frame[:row]
        #         elif row > frame.shape[0]:
        #             temp = np.zeros((row-frame.shape[0], frame.shape[1]))
        #             frame = np.vstack((frame, temp))
        #     features.append(frame)
        #     labels.append(l)

    # print(np.shape(features))
    # print(np.shape(features[0]))
    return np.array(features), np.array(labels)


def extract_test(_paths):
    features = []
    for p, l in zip(_paths,):
        print(p)

        try:
            if p[-1] == '\n':
                y, sr = lb.load(p[:-1], res_type='kaiser_fast')
            else:
                y, sr = lb.load(p, res_type='kaiser_fast')
            duration = y.shape[0] / sr
            parts = int(duration / part_in_seconds)
            part_len = sr*part_in_seconds
        except AssertionError:
            continue
        except RuntimeError:
            continue

        for i in range(0, parts*part_len, part_len):
            mfc = lb.feature.melspectrogram(y=y[i:i+part_len], sr=sr).T
            log_S = lb.logamplitude(mfc, ref_power=np.max)
            features.append(log_S)

    return np.array(features)


def extract(_paths, _labels=None):

    if _labels is None:
        return extract_test(_paths)
    else:
        return extract_train(_paths, _labels)

""""
x_train, y_train, x_val, y_val, x_test, y_test = read_instructions()
train_set = data_set(x_train, y_train)
xx, yy = subsetn_random(train_set)
f, l = extract(xx, yy)
"""
def extract_n(train_set):
    xx, yy = subsetn_random(train_set)
    f, l = extract(xx, yy)
    return f, l
