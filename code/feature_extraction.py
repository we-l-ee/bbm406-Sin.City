import librosa as lb
import numpy as np
import time
import os
from sklearn import preprocessing
from random import shuffle
# root_dir = sys.argv[1]
# max_part = 5
# encoded_labels = {'akdeniz': 0, 'doguanadolu': 1, 'ege': 2, 'guneydoguanadolu': 3, 'icanadolu': 4, 'karadeniz': 5,
# 'trakya': 6}

test_percent = 25
train_percent = 75

scaler = preprocessing.StandardScaler()

data_dir = "data/"
train_data_dir = data_dir+'train/'
test_data_dir = data_dir+'test/'
all_data_dir = data_dir+'all_data/'

if not os.path.exists(data_dir):
    os.mkdir(data_dir)
if not os.path.exists(train_data_dir):
    os.mkdir(train_data_dir)
if not os.path.exists(test_data_dir):
    os.mkdir(test_data_dir)
if not os.path.exists(all_data_dir):
    os.mkdir(all_data_dir)


def compute_melspectogram(y, **kwargs):
    if 'n_mels' not in kwargs:
        kwargs['n_mels'] = 60           # band
    melspec = lb.feature.melspectrogram(y, n_mels=kwargs['n_mels'])
    log_S = lb.logamplitude(melspec).T
    return log_S


def compute_mfcc(y, **kwargs):
    if 'n_mfcc' not in kwargs:
        kwargs['n_mfcc'] = 60
    mfcc = lb.feature.mfcc(y, n_mfcc=kwargs['n_mfcc'])
    return mfcc


def compute_spectral_centroid(y, sr, **kwargs):
    centroid = lb.feature.spectral_centroid(y, sr=sr)
    return centroid


def compute_spectral_rolloff(y, **kwargs):
    roll_off = lb.feature.spectral_rolloff(y)
    return roll_off


def zero_crossing_rate(y, sr, **kwargs):
    centroid = lb.feature.zero_crossing_rate(y)
    return centroid


compute_feature = {
    'melspectogram': compute_melspectogram,
    'mfcc': compute_mfcc,
    'zero_crossing_rate': zero_crossing_rate,
    'spectral_rolloff': compute_spectral_rolloff,
    'spectral_centroid': compute_spectral_centroid
}


def shuffle_list(*ls):
    l = list(zip(*ls))
    shuffle(l)
    return zip(*l)


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


def subsetn_random(set, train_subsetn=10):
    x, y = [], []
    np.random.seed(int(time.time()))
    for l in set:
        xx = np.array(set[l], dtype=object)
        np.random.shuffle(xx)
        xx = xx[:train_subsetn]
        yy = [l for _ in range(train_subsetn)]
        x.extend(xx)
        y.extend(yy)

    x, y = shuffle_list(x, y)
    return np.array(x), np.array(y, dtype=int)


def data_set(xx, yy):
    set = dict()
    for x, y in zip(xx, yy):
        try:
            set[y].append(x)
        except KeyError:
            set[y] = [x]
    return set


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)


def extract_features(paths, labels_, extract_type, batch=100, feature_type='melspectogram', seconds=30, frames=41,
                     **kwargs):

    global parts_in_song
    parts_in_song = 0

    window_size = 512 * (frames - 1)
    features = []
    labels = []
    count = 0
    name = 0
    for label, path in zip(labels_, paths):
        try:
            if path[-1] == '\n':
                y, sr = lb.load(path[:-1], res_type='kaiser_fast', duration=seconds)
            else:
                y, sr = lb.load(path, res_type='kaiser_fast', duration=seconds)
        except AssertionError:
            continue
        print('Extracting', path)
        for start, end in windows(y, window_size):
            # print(start, end)
            if len(y[start:end]) == int(window_size):
                signal = y[start:end]

                compute_func = compute_feature[feature_type]
                feature = compute_func(signal, **kwargs)
                scaled_feature = scaler.fit_transform(feature)
                features.append(scaled_feature)
                # features.append(feature)
                labels.append(label)
                parts_in_song += 1

        count += 1
        if count == batch:
            if extract_type == 'train':
                save_data(name, features, labels, train_data_dir)
            elif extract_type == 'test':
                save_data(name, features, labels, test_data_dir)
            features = []
            labels = []
            count = 0
            name += 1

    if len(labels) > 0:
        if extract_type == 'train':

            if batch == 'all':
                save_data('train', features, np.array(labels, dtype=np.int))
            else:
                save_data(name, features, np.array(labels, dtype=np.int), train_data_dir)
        elif extract_type == 'test':

            if batch == 'all':
                save_data('test', features, np.array(labels, dtype=np.int))
            else:
                save_data(name, features, np.array(labels, dtype=np.int), test_data_dir)

parts = 63 #parts_in_song


def save_data(name, features, labels, data_dir=all_data_dir):

    file_name = str(name)
    print("Saving " + file_name)

    # print("Features of", fold_name, " = ", features.shape)
    # print("Labels of", fold_name, " = ", labels.shape)

    feature_file = os.path.join(data_dir, file_name + '_x.npy')
    labels_file = os.path.join(data_dir, file_name + '_y.npy')
    np.save(feature_file, features)
    np.save(labels_file, labels)
    print("Saved " + feature_file)
    print("Saved " + labels_file)


def gather_batches(batch_type='train'):
    features, labels = [], []
    data_dir = None

    if batch_type == 'train':
        data_dir = train_data_dir

    elif batch_type == 'test':
        data_dir = test_data_dir

    for dir_name, subdir_list, file_list in os.walk(data_dir):
        for f_name in file_list:
            type = str(f_name.split("_")[1].split('.')[0])
            file_path = os.path.join(dir_name, f_name)
            if type == 'x':
                features.append(file_path)
            elif type == 'y':
                labels.append(file_path)

    return features, labels


def load_data(name, data_dir=all_data_dir):

    feature_file = os.path.join(data_dir, name + '_x.npy')
    labels_file = os.path.join(data_dir, name + '_y.npy')
    loaded_features = np.load(feature_file)
    loaded_labels = np.load(labels_file)
    print(name, "features: ", loaded_features.shape)

    return loaded_features, loaded_labels

"""
part_in_seconds = 10
encoded_labels = dict()
label_counter = 0
train_subsetn = 10
from version1


def data_load(audio_f, sr=22050, file_format="wav", num_channels=1):
    audio_binary = tf.read_file(audio_f)
    y = tf.contrib.ffmpeg.decode_audio(audio_binary, file_format, sr, num_channels)
    return tf.squeeze(y, 1), sr


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


"""