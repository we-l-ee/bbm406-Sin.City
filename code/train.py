import argparse
import os

import feature_extraction as feature

import numpy as np
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import np_utils

from sklearn import svm
cnn_models_path = 'model/cnn_models/'



def build_model(X=None, nb_classes=7, nb_layers=2):
    filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size
    input_shape = (X.shape[1], X.shape[2], 1)

    model = Sequential()
    model.add(Conv2D(filters, kernel_size, padding='valid', input_shape=input_shape))

    model.add(BatchNormalization())
    model.add(Activation('relu'))

    for layer in range(nb_layers - 1):
        model.add(Conv2D(filters, kernel_size))
        model.add(BatchNormalization())
        model.add(ELU(alpha=1.0))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


def load_cnn_models(cnn_models_path):
    cnn_models = []
    for dir_name, subdir_list, file_list in os.walk(cnn_models_path):
        for f_name in file_list:
            file_path = os.path.join(dir_name, f_name)
            cnn_models.append(load_model(file_path))

    return cnn_models


def save_cnn_model(cnn_models_path, cnn_models):
    if not os.path.exists(cnn_models_path):
        os.mkdir(cnn_models_path)

    i = 0
    for cnn in cnn_models:
        cnn.save(cnn_models_path+str(i)+'.h5')
    i += 0



def main(cnn_number=4, train_subset=2, fit_time_per_model=1):

    x_train, y_train, x_test, y_test = feature.read_instructions()
    train_set = feature.data_set(x_train, y_train)

    print('data preprocessing is completed! Starts to train')
    # test_set = feature.data_set(x_test, y_test)
    train(train_set, cnn_number, train_subset, fit_time_per_model)
    print('Training is completed, loading models')
    cnn_models = load_cnn_models(cnn_models_path)

    print('loading models completed, test is starting ')
    exit(0)
    test(x_test, y_test, cnn_models)
    print('test is completed')


def train(train_set, cnn_number, train_subset, fit_time_per_model):
    cnn_models = list()
    batch_size = 16
    epochs = 5

    for _ in range(cnn_number):
        xt, yt = feature.subsetn_random(train_set, train_subset)
        ft, lt = feature.extract(xt, yt)
        print('creating model', _)
        x_train = ft.reshape(ft.shape[0], ft.shape[1], ft.shape[2], 1).astype('float32')
        nb_classes = len(np.unique(lt))
        y_train = np_utils.to_categorical(lt, nb_classes)

        model = build_model(x_train, nb_classes=nb_classes)

        for fit_t in range(fit_time_per_model):
            if fit_t != 0:
                xt, yt = feature.subsetn_random(train_set, train_subset)
                ft, lt = feature.extract(xt, yt)
                x_train = ft.reshape(ft.shape[0], 1, ft.shape[1], ft.shape[2]).astype('float32')
                y_train = np_utils.to_categorical(lt, len(np.unique(lt)))

            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1)

        cnn_models.append(model)
    save_cnn_model(cnn_models_path, cnn_models)


def estimatep(pred):
    labels, counts = pred

    index = np.argmax(counts)
    sum_ = np.sum(counts)
    percent = (counts[index]/sum_)*100
    label = labels[index]

    print(counts, labels)
    print(index)
    print(counts[index])
    print('label:', label)
    print(percent)
    print('===============')
    return label, percent


def test(x_test, y_test, cnn_models, process='accuracy-percent', **kwargs):

    def default():
        acc, total = 0, 0
        def estimations(models):
            votes = []
            for pred in models:
                label, percent = pred
                if percent > kwargs['percent']:
                    pass

    model_estimations = [predict(x_test, model, "estimate-percent") for model in cnn_models]
    print(model_estimations)
    exit(0)
    for estimations in model_estimations:
        for estimate in estimations:
            label, percemt = estimate


def predict(x_test, models, process='all-predictions'):
    def default(predicted_labels):
        return np.unique(predicted_labels, return_counts=True)

    all_func = {'all-predictions': default, 'estimate-percent': estimatep}
    func = all_func[process]

    preds = []
    for path in x_test:
        each_model = []
        for model in models:
            f = feature.extract([path])
            x_test = f.reshape(f.shape[0], f.shape[1], f.shape[2], 1).astype('float32')

            each_model.append(func(model.predict(x_test)))

        preds.append(each_model)

    return preds
''''
x_train, y_train, x_test, y_test = feature.read_instructions()
train_set = feature.data_set(x_train, y_train)
xt, yt = feature.subsetn_random(train_set, 1)
ft, lt = feature.extract(xt, yt)
x_train = ft.reshape(ft.shape[0], ft.shape[1], ft.shape[2], 1).astype('float32')
'''''
main(cnn_number=2, train_subset=2, fit_time_per_model=1)

