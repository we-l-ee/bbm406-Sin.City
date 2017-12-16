import librosa as lb
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from tinytag import TinyTag
import audioread
import magic

import sklearn

# 2 random offset by 30 seconds
sample_length = 660000
hamming_size = 1000
hamming_stride = 1000

root_dir = 'dataset'
features = []
labels = list()
part_in_seconds = 30
max_part = 5
# encoded_labels = {'akdeniz': 0, 'doguanadolu': 1, 'ege': 2, 'guneydoguanadolu': 3, 'icanadolu': 4, 'karadeniz': 5, 'trakya': 6}
encoded_labels = dict()
label_counter = 0
for dir_name, subdir_list, file_list in os.walk(root_dir):
    for f_name in file_list:
        file_path = os.path.join(dir_name, f_name)
        print(file_path)
        label = file_path.split('\\')[1]

        with audioread.audio_open(file_path) as f:
            duration = f.duration

        parts = duration//part_in_seconds
        if parts > max_part:
            parts = max_part

        y, sr = lb.load(file_path, duration=part_in_seconds*parts)
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

if not os.path.exists('model'):
    os.makedirs('model')

np.save('model/log_melspectrogram.npy', features)
np.save('model/labels.npy', labels)
np.save('model/encoded_labels.npy', [encoded_labels])


