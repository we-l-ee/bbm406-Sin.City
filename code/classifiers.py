import numpy as np
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import np_utils

from sklearn import svm
from sklearn import neighbors
from sklearn.externals import joblib

import feature_extraction as feature
from abc import ABC, abstractmethod


class BaseClassifier(ABC):

    models_path = None
    model = None
    name = ''

    @abstractmethod
    def build_model(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, x_train, y_train, **kwargs):
        pass

    @abstractmethod
    def prepare_data(self, ft, lt):
        pass

    @abstractmethod
    def load(self, file_path):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def predict(self, x_test, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, predictions, y_test):
        pass

    def train(self, train_set, train_subset, fit_time_per_model, feature_type, **kwargs):
        xt, yt = feature.subsetn_random(train_set, train_subset)
        ft, lt = feature.extract(xt, yt, feature_type)
        print('creating model', self.name)
        nb_classes = len(np.unique(lt))

        x_train, y_train = self.prepare_data(ft, lt)
        self.build_model(x_train, nb_classes=nb_classes)

        for fit_t in range(fit_time_per_model):
            if fit_t != 0:
                x_train, y_train = self.prepare_data(ft, lt)

            self.fit(x_train, y_train, **kwargs)


class CNNClassifier(BaseClassifier):

    models_path = 'models/cnn_models/'
    model = None
    name = 'cnn'

    def save(self, number):
        self.model.save(self.models_path + str(number) + '.h5')

    def load(self, file_path):
        self.model = load_model(file_path)

    def prepare_data(self, ft, lt):
        x_train, y_train = None, None
        if ft is not None:
            x_train = ft.reshape(ft.shape[0], ft.shape[1], ft.shape[2], 1).astype('float32')
        if lt is not None:
            nb_classes = len(np.unique(lt))
            y_train = np_utils.to_categorical(lt, nb_classes)
        return x_train, y_train

    def fit(self, x_train, y_train, **kwargs):
        self.model.fit(x_train, y_train,
                       batch_size=kwargs['batch_size'],
                       epochs=kwargs['epoch'])

    def build_model(self, x_train=None, nb_classes=7, nb_layers=2):

        filters = 32  # number of convolutional filters to use
        pool_size = (2, 2)  # size of pooling area for max pooling
        kernel_size = (3, 3)  # convolution kernel size
        input_shape = (x_train.shape[1], x_train.shape[2], 1)

        self.model = Sequential()
        self.model.add(Conv2D(filters, kernel_size, input_shape=input_shape))

        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        for layer in range(nb_layers - 1):
            self.model.add(Conv2D(filters, kernel_size))
            self.model.add(BatchNormalization())
            self.model.add(ELU(alpha=1.0))
            self.model.add(MaxPooling2D(pool_size=pool_size))
            self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation("softmax"))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adadelta',
                           metrics=['accuracy'])

    def predict(self, x_test, **kwargs):
        if 'batch_size' not in kwargs['batch_size']:
            kwargs['batch_size'] = 32
        return self.model.predict(x_test, batch_size=kwargs['batch_size'], verbose=1)

    def evaluate(self, predictions, y_test):
        pass


class SVMClassifier(BaseClassifier):

    models_path = 'models/svm_models/'
    model = None
    name = 'svm'

    def save(self, number):
        joblib.dump(self.model, self.models_path + str(number) + '.pkl')

    def load(self, file_path):
        self.model = joblib.load(file_path)

    def build_model(self):
        model = svm.LinearSVC()
        self.model = model

    def prepare_data(self, ft, lt):
        x_train, y_train = None, None
        if ft is not None:
            x_train = ft.reshape(ft.shape[0], ft.shape[1] + ft.shape[2]).astype('float32')
        if lt is not None:
            y_train = lt
        return x_train, y_train

    def fit(self, x_train, y_train, **kwargs):
        self.model.fit(x_train, y_train)

    def predict(self, x_test, **kwargs):
        return self.model.predict(x_test)

    def evaluate(self, predictions, y_test):
        pass


class KNNClassifier(BaseClassifier):

    models_path = 'models/knn_models/'
    name = 'knn'
    model = None

    def save(self, number):
        pass

    def load(self, file_path):
        pass

    def build_model(self):
        model = neighbors.KNeighborsClassifier()
        self.model = model

    def prepare_data(self, ft, lt):
        x_train, y_train = None, None
        if ft is not None:
            x_train = ft.reshape(ft.shape[0], ft.shape[1] + ft.shape[2]).astype('float32')
        if lt is not None:
            y_train = lt

        return x_train, y_train

    def fit(self, x_train, y_train, **kwargs):
        self.model.fit(x_train, y_train)

    def predict(self, x_test, **kwargs):
        return self.model.predict(x_test)

    def evaluate(self, predictions, y_test):
        pass

mappings = {'svm': SVMClassifier(), 'cnn': CNNClassifier(), 'knn': KNNClassifier()}


def get_model(str_list):
    classifiers = list()
    for str in str_list:
        classifiers.append(mappings[str])
    return classifiers

