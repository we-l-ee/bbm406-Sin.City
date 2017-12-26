import librosa as lb
import numpy as np
import time
import glob
import os
from sklearn import preprocessing
import tensorflow as tf
# root_dir = sys.argv[1]
# max_part = 5
# encoded_labels = {'akdeniz': 0, 'doguanadolu': 1, 'ege': 2, 'guneydoguanadolu': 3, 'icanadolu': 4, 'karadeniz': 5,
# 'trakya': 6}
part_in_seconds = 10
encoded_labels = dict()
label_counter = 0
train_subsetn = 10


test_percent = 25
train_percent = 75

scaler = preprocessing.StandardScaler()


def data_load(audio_f, sr=22050, file_format="wav", num_channels=1):
    audio_binary = tf.read_file(audio_f)
    y = tf.contrib.ffmpeg.decode_audio(audio_binary, file_format, sr, num_channels)
    return tf.squeeze(y, 1), sr


def compute_melspectogram(y, sr, i, part_len):
    mfc = lb.feature.melspectrogram(y=y[i:i + part_len], sr=sr).T
    log_S = lb.logamplitude(mfc, ref_power=np.max)
    return log_S


def compute_mfcc(y, sr, i, part_len):
    mfcc = lb.feature.mfcc(y=y[i:i + part_len], sr=sr, n_mfcc=part_in_seconds*4)
    return mfcc


def compute_spectral_centroid(y, sr, i, part_len):
    centroid = lb.feature.spectral_centroid(y=y[i:i + part_len], sr=sr)
    return centroid


def compute_spectral_rolloff(y, sr, i, part_len):
    roll_off = lb.feature.spectral_rolloff(y=y[i:i + part_len], sr=sr)
    return roll_off


def zero_crossing_rate(y, sr, i, part_len):
    centroid = lb.feature.zero_crossing_rate(y=y[i:i + part_len])
    return centroid


compute_feature = {
    'melspectogram': compute_melspectogram,
    'mfcc': compute_mfcc,
    'zero_crossing_rate': zero_crossing_rate,
    'spectral_rolloff': compute_spectral_rolloff,
    'spectral_centroid': compute_spectral_centroid
}


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
    return np.array(x), np.array(y, dtype=int)


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
        yy = [l for i in range(train_subsetn)]
        x.extend(xx)
        y.extend(yy)
    return np.array(x), np.array(y, dtype=int)


def extract_train(_paths, _labels, feature_t):

    labels = []
    features = []

    for p, l in zip(_paths, _labels):
        print(p)

        try:
            if p[-1] == '\n':
                # format = p[:-1].split('.')[-1]
                print(format, p[:-1])
                # y, sr = data_load(p[:-1], 22050, format, 1)
                y, sr = lb.load(p[:-1], res_type='kaiser_fast')
            else:
                # format = p.split('.')[-1]
                # print(format, p)
                # y, sr = data_load(p, 22050, format, 1)
                y, sr = lb.load(p, res_type='kaiser_fast')
            duration = y.shape[0] / sr
            parts = int(duration / part_in_seconds)
            part_len = sr*part_in_seconds
        except AssertionError:
            continue
        except RuntimeError:
            continue

        for i in range(0, parts*part_len, part_len):

            compute_func = compute_feature[feature_t]
            feature = compute_func(y, sr, i, part_len)
            scaled_feature = scaler.fit_transform(feature)
            features.append(scaled_feature)
            labels.append(l)

    return np.array(features), np.array(labels)


def extract_test(_paths, feature_t):
    features = []
    for p in _paths:
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

        for i in range(0, parts*part_len, part_len):
            compute_func = compute_feature[feature_t]
            feature = compute_func(y, sr, i, part_len)
            scaled_feature = scaler.fit_transform(feature)
            features.append(scaled_feature)

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
    return np.array(features)


def extract(_paths, _labels=None, feature_t='melspectogram'):
    if _labels is None:
        return extract_test(_paths, feature_t)
    else:
        return extract_train(_paths, _labels, feature_t)

""""
x_train, y_train, x_val, y_val, x_test, y_test = read_instructions()
train_set = data_set(x_train, y_train)
xx, yy = subsetn_random(train_set)
f, l = extract(xx, yy)



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
    def extract_n(train_set):
    xx, yy = subsetn_random(train_set)
    f, l = extract(xx, yy)
    return f, l
"""


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)


def extract_features(paths, labels_ , bands=60, frames=41):
    window_size = 512 * (frames - 1)
    log_specgrams = []

    labels = []
    for label, path in zip(labels_, paths):
        try:
            if p[-1] == '\n':
                y, sr = lb.load(path[:-1], res_type='kaiser_fast')
            else:
                y, sr = lb.load(path, res_type='kaiser_fast')
        except AssertionError:
            continue

        for start, end in window_size(y, window_size):
            if len(sound_clip[start:end]) == int(window_size):
                signal = y[start:end]
                melspec = lb.feature.melspectrogram(signal, n_mels=bands)
                logspec = lb.logamplitude(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)
                labels.append(label)

    return np.array(features), np.array(labels, dtype=np.int)


def save_data(fold_k, features, labels,data_dir='data/'):

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    fold_name = 'fold' + str(fold_k)
    print("\\nSaving " + fold_name)

    # print("Features of", fold_name, " = ", features.shape)
    # print("Labels of", fold_name, " = ", labels.shape)

    feature_file = os.path.join(data_dir, fold_name + '_x.npy')
    labels_file = os.path.join(data_dir, fold_name + '_y.npy')
    np.save(feature_file, features)
    print("Saved " + feature_file)
    np.save(labels_file, labels)
    print("Saved " + labels_file)



# this will aggregate all the training data
def load_all_folds(data_dir, features, labels):
    subsequent_fold = False
    for k in range(1, 9):
        fold_name = 'fold' + str(k)
        print("\nAdding " + fold_name)
        feature_file = os.path.join(data_dir, fold_name + '_x.npy')
        labels_file = os.path.join(data_dir, fold_name + '_y.npy')
        loaded_features = np.load(feature_file)
        loaded_labels = np.load(labels_file)
        print("New Features: ", loaded_features.shape)

        if subsequent_fold:
            train_x = np.concatenate((features, loaded_features))
            train_y = np.concatenate((labels, loaded_labels))
        else:
            train_x = loaded_features
            train_y = loaded_labels
            subsequent_fold = True

    # use the penultimate fold for validation
    valid_fold_name = 'fold9'
    feature_file = os.path.join(data_dir, valid_fold_name + '_x.npy')
    labels_file = os.path.join(data_dir, valid_fold_name + '_y.npy')
    valid_x = np.load(feature_file)
    valid_y = np.load(labels_file)

    # and use the last fold for testing
    test_fold_name = 'fold10'
    feature_file = os.path.join(data_dir, test_fold_name + '_x.npy')
    labels_file = os.path.join(data_dir, test_fold_name + '_y.npy')
    test_x = np.load(feature_file)
    test_y = np.load(labels_file)


# this is used to load the folds incrementally
def load(folds, data_dir="data/"):
    subsequent_fold = False
    for k in range(len(folds)):
        fold_name = 'fold' + str(folds[k])
        feature_file = os.path.join(data_dir, fold_name + '_x.npy')
        labels_file = os.path.join(data_dir, fold_name + '_y.npy')
        loaded_features = np.load(feature_file)
        loaded_labels = np.load(labels_file)
        print(fold_name, "features: ", loaded_features.shape)

        if subsequent_fold:
            features = np.concatenate((features, loaded_features))
            labels = np.concatenate((labels, loaded_labels))
        else:
            features = loaded_features
            labels = loaded_labels
            subsequent_fold = True

    return features, labels
