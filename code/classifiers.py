import numpy as np
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras import backend as K
from keras import optimizers

# from keras.callbacks import EarlyStopping
# from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn import linear_model # SGD with SVM
from sklearn import neighbors
from sklearn.externals import joblib
from sklearn import metrics


import feature_extraction as feature
from abc import ABC, abstractmethod

import os

counter = 0
all_models_path = 'models'

if not os.path.exists(all_models_path):
    os.mkdir(all_models_path)


def save_all_models(models):
    i = 0
    for classifier in models:
        classifier.save(i)
        i += 1


def evaluate_accuracy(predictions, y_test, f1_score=False):
    if f1_score:
        return metrics.f1_score(y_test, predictions, average='micro')
    return metrics.accuracy_score(y_test, predictions)


def load_models(path='models/'):
    models = list()
    for dir_name, subdir_list, file_list in os.walk(path):
        for f_name in file_list:
            file_path = os.path.join(dir_name, f_name)
            model = mappings[dir_name.split('_')[0].split('\\')[1]]    # create empty model
            model.load(file_path)                           # load model
            models.append(model)
    return models


class BaseClassifier(ABC):

    models_path = None
    model = None
    name = ''

    @abstractmethod
    def build_model(self, *args, **kwargs):
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
    def save(self):
        pass

    @abstractmethod
    def predict(self, x_test, **kwargs):
        pass

    ''''
    from version .1
    def train(self, train_set, train_subset, fit_time_per_model, feature_type, **kwargs):
        xt, yt = feature.subsetn_random(train_set, train_subset)
        ft, lt = feature.extract(xt, yt, feature_type)
        print('creating model', self.name)

        x_train, y_train = self.prepare_data(ft, lt)

        self.build_model(x_train, y_train, **kwargs)

        for fit_t in range(fit_time_per_model):
            if fit_t != 0:
                x_train, y_train = self.prepare_data(ft, lt)

            self.fit(x_train, y_train, **kwargs)
        self.save()
        K.clear_session()
    '''

    def show(self):
        self.model.show()

    def train_normal(self, **kwargs):
        print('INFO: Loading data!')
        x_train, y_train = feature.load_data('train')
        print('INFO: Preparing data for fitting!')
        x_train, y_train = self.prepare_data(x_train, y_train)
        print('INFO: Creating model!')
        self.build_model(x_train, y_train, **kwargs)
        print('INFO: Fitting model!')
        self.fit(x_train, y_train,**kwargs)
        print('INFO: Saving model!')
        self.save()
        K.clear_session()

    def train_batch(self, **kwargs):
        x_batches_path, y_batches_path = feature.gather_batches()
        number = 0
        for x_path, y_path in zip(x_batches_path, y_batches_path):

            print('INFO: Loading data! batch number:', number)
            x_train = np.load(x_path)
            y_train = np.load(y_path)
            print('INFO: Preparing data for fitting!')
            x_train, y_train = self.prepare_data(x_train, y_train)
            if number == 0:
                print('INFO: Creating model!')
                self.build_model(x_train, y_train, **kwargs)
            print('INFO: Fitting model!')
            self.fit(x_train, y_train, **kwargs)
            number += 1
        self.save()
        K.clear_session()

    def train(self, train_type, **kwargs):

        if train_type == 'all':
            self.train_normal(**kwargs)
        elif train_type == 'batch':
            if self.name == 'knn':
                print('INFO: KNN can not be used with batch learning! Changing to normal train')
                self.train_normal(**kwargs)
            else:
                self.train_batch(**kwargs)

    def estimatep(self, pred):

        labels, counts = np.unique(pred, return_counts=True)

        index = np.argmax(counts)
        sum_ = np.sum(counts)
        percent = (counts[index] / sum_) * 100
        label = labels[index]

        return label, percent

    def test_batch(self, **kwargs):
        x_batches_path, y_batches_path = feature.gather_batches('test')
        number = 0
        all_preds = []
        real_labels = []
        for x_path, y_path in zip(x_batches_path, y_batches_path):

            #if number == 2:
            #    break
            print('INFO: Loading data! batch number:', number)
            x_test = np.load(x_path)
            y_test = np.load(y_path)
            print('INFO: Preparing data for testing!')
            x_test, y_test = self.prepare_data(x_test, y_test)
            print('INFO: Testing model!')
            preds = self.predict(x_test, **kwargs)
            parts_in_song = feature.parts
            songs_preds = preds.reshape(-1, parts_in_song)
            songs_labels = y_test.reshape(-1, parts_in_song)

            preds = np.array([self.estimatep(song_pred) for song_pred in songs_preds])
            all_preds.extend(preds)
            real_labels.extend(songs_labels[:,0])
            print(preds)

            number += 1

        all_preds = np.array(all_preds)
        return np.array(all_preds), np.array(real_labels)

    def test_normal(self, **kwargs):
        print('INFO: Loading data!')
        x_test, y_test = feature.load_data('test')
        print('INFO: Preparing data for testing!')
        x_test, y_test = self.prepare_data(x_test, y_test)
        y_test = np.array([np.argmax(l) for l in y_test])
        print('INFO: Testing model!')
        preds = self.predict(x_test, **kwargs)
        parts_in_song = feature.parts
        songs_preds = preds.reshape(-1, parts_in_song)
        songs_labels = y_test.reshape(-1, parts_in_song)
        preds = np.array([self.estimatep(song_pred) for song_pred in songs_preds])

        return preds, songs_labels[:, 0]

    def test(self, test_type, **kwargs):

        if test_type == 'all':
            return self.test_normal(**kwargs)
        elif test_type == 'batch':
            if self.name == 'knn':
                print('INFO: KNN can not be used with batch testing! Changing to normal test')
                return self.test_normal(**kwargs)
            else:
                return self.test_batch(**kwargs)

        # return preds



class NNClassifier(BaseClassifier):

    models_path = 'models/nn_models/'
    model = None
    name = 'nn'

    if not os.path.exists(models_path):
        os.mkdir(models_path)

    def save(self):
        self.model.save(self.models_path + str(self.__hash__()) + '.h5')

    def load(self, file_path):
        self.model = load_model(file_path)

    def prepare_data(self, ft, lt):
        x_train, y_train = None, None
        if ft is not None:
            x_train = ft.reshape(ft.shape[0], ft.shape[1] * ft.shape[2]).astype('float32')
        if lt is not None:
            labels = list(np.load('model/labels.npy')[0].values())
            nb_classes = len(np.unique(labels))
            y_train = np_utils.to_categorical(lt, nb_classes)
        return x_train, y_train

    def fit(self, x_train, y_train, **kwargs):

        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = 32
        if 'epoch' not in kwargs:
            kwargs['epoch'] = 10

        self.model.fit(x_train, y_train,
                       batch_size=kwargs['batch_size'],
                       epochs=kwargs['epoch'])

    def build_model(self, *args, **kwargs):
        nn_layers = kwargs['nn_layers']
        self.model = Sequential()

        for i in range(len(nn_layers)):
            if i == 0:
                self.model.add(Dense(nn_layers[i], input_dim=args[0].shape[1], activation='relu'))
            else:
                self.model.add(Dense(nn_layers[i], activation='sigmoid'))
        self.model.add(Dense(args[1].shape[1], activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])

    def predict(self, x_test, **kwargs):
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = 32
        return self.model.predict(x_test, batch_size=kwargs['batch_size'], verbose=1)


class CNNClassifier(BaseClassifier):

    models_path = 'models/cnn_models/'
    model = None
    name = 'cnn'

    if not os.path.exists(models_path):
        os.mkdir(models_path)
    
    def save(self):
        self.model.save(self.models_path + str(self.__hash__()) + '.h5')

    def load(self, file_path):
        self.model = load_model(file_path)

    def prepare_data(self, ft, lt):
        x_train, y_train = None, None
        if ft is not None:
            x_train = ft.reshape(ft.shape[0], ft.shape[2], ft.shape[1], 1).astype('float32')
        if lt is not None:
            labels = list(np.load('model/labels.npy')[0].values())
            nb_classes = len(np.unique(labels))
            y_train = np_utils.to_categorical(lt, nb_classes)
        return x_train, y_train

    def fit(self, x_train, y_train, **kwargs):

        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = 64
        if 'epochs' not in kwargs:
            kwargs['epochs'] = 5

        self.model.fit(x_train, y_train,
                       batch_size=kwargs['batch_size'],
                       epochs=kwargs['epochs'], verbose=1)
    """
    
    from version .1
    def build_model(self, *args, **kwargs):

        filters = 32  # number of convolutional filters to use
        pool_size = (2, 2)  # size of pooling area for max pooling
        kernel_size = (3, 3)  # convolution kernel size
        input_shape = (args[0].shape[1], args[0].shape[2], 1)

        self.model = Sequential()
        self.model.add(Conv2D(filters, kernel_size, input_shape=input_shape))

        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        if 'nb_layers' not in kwargs:
            kwargs['nb_layers'] = 2
        for layer in range(kwargs['nb_layers'] - 1):
            self.model.add(Conv2D(filters, kernel_size))
            self.model.add(BatchNormalization())
            self.model.add(ELU(alpha=1.0))
            self.model.add(MaxPooling2D(pool_size=pool_size))
            self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(args[1].shape[1]))
        self.model.add(Activation("softmax"))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adadelta',
                           metrics=['accuracy'])
    """

    def build_model(self, *args, **kwargs):

        filters = 48 # number of convolutional filters to use
        # pool_size = (2, 2)  # size of pooling area for max pooling
        kernel_size = (1, 1)  # convolution kernel size
        input_shape = (args[0].shape[1], args[0].shape[2], 1)


        # first layer has 48 convolution filters
        self.model = Sequential()
        self.model.add(Conv2D(96, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(96, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(96, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))


        self.model.add(Flatten())
        self.model.add(Dense(90, activation='relu'))
        self.model.add(Dense(90, activation='relu'))
        # output layer
        self.model.add(Dense(args[1].shape[1], activation='softmax'))


        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    def predict(self, x_test, **kwargs):
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = 32
        preds = self.model.predict(x_test, batch_size=kwargs['batch_size'], verbose=1)
        return np.array([np.argmax(pred) for pred in preds])


class SVMClassifier(BaseClassifier):

    models_path = 'models/svm_models/'
    model = None
    name = 'svm'

    if not os.path.exists(models_path):
        os.mkdir(models_path)

    def save(self):
        joblib.dump(self.model, self.models_path + str(self.__hash__()) + '.pkl')

    def load(self, file_path):
        self.model = joblib.load(file_path)

    def build_model(self, *args, **kwargs):
        model = linear_model.SGDClassifier(loss="hinge", penalty="l2", verbose=1, max_iter=100)
        self.model = model

    def prepare_data(self, ft, lt):
        x_train, y_train = None, None
        if ft is not None:
            x_train = ft.reshape(ft.shape[0], ft.shape[1] * ft.shape[2]).astype('float32')
        if lt is not None:
            y_train = lt
        return x_train, y_train

    def fit(self, x_train, y_train, **kwargs):
        labels = list(np.load('model/labels.npy')[0].values())
        self.model.partial_fit(x_train, y_train, classes=labels)

    def predict(self, x_test, **kwargs):
        return self.model.predict(x_test)


class KNNClassifier(BaseClassifier):

    models_path = 'models/knn_models/'
    name = 'knn'
    model = None

    if not os.path.exists(models_path):
        os.mkdir(models_path)
    
    def save(self):
        pass

    def load(self, file_path):
        pass

    def build_model(self, *args, **kwargs):
        model = neighbors.KNeighborsClassifier()
        self.model = model

    def prepare_data(self, ft, lt):
        x_train, y_train = None, None
        if ft is not None:
            x_train = ft.reshape(ft.shape[0], ft.shape[1] * ft.shape[2]).astype('float32')
        if lt is not None:
            y_train = lt

        return x_train, y_train

    def fit(self, x_train, y_train, **kwargs):
        self.model.fit(x_train, y_train)

    def predict(self, x_test, **kwargs):
        return self.model.predict(x_test)


mappings = {'svm': SVMClassifier(), 'cnn': CNNClassifier(), 'knn': KNNClassifier(), 'nn': NNClassifier()}


def get_model(str_list):
    classifiers = list()
    for str in str_list:
        classifiers.append(mappings[str])
    return classifiers


'''
will be imported

def build_model_v2():

    frames = 41
    bands = 60
    feature_size = bands * frames  # 60x41
    num_labels = 10
    num_channels = 2
    model = Sequential()
    # input: 60x41 data frames with 2 channels => (60,41,2) tensors

    # filters of size 3x3 - paper describes using 5x5, but their input data is 128x128
    f_size = 3

    # Layer 1 - 24 filters with a receptive field of (f,f), i.e. W has the
    # shape (24,1,f,f).  This is followed by (4,2) max-pooling over the last
    # two dimensions and a ReLU activation function
    model.add(
        Conv2D(24, f_size, f_size, border_mode='same', init="normal", input_shape=(bands, frames, num_channels)))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation('relu'))

    # Layer 2 - 48 filters with a receptive field of (f,f), i.e. W has the
    # shape (48, 24, f, f). Like L1 this is followed by (4,2) max-pooling
    # and a ReLU activation function.
    model.add(Conv2D(48, f_size, f_size, init="normal", border_mode='same'))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation('relu'))

    # Layer 3 - 48 filters with a receptive field of (f,f), i.e. W has the
    # shape (48, 48, f, f). This is followed by a ReLU but no pooling.
    model.add(Conv2D(48, f_size, f_size, border_mode='valid'))
    model.add(Activation('relu'))

    # flatten output into a single dimension, let Keras do shape inference
    model.add(Flatten())

    # Layer 4 - a fully connected NN layer of 64 hidden units, L2 penalty of 0.001
    model.add(Dense(64, W_regularizer=optimizers.l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Layer 5 - an output layer with one output unit per class, with L2 penalty,
    # followed by a softmax activation function
    model.add(Dense(num_labels, W_regularizer=optimizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Activation('softmax'))

    # compile and fit model, reduce epochs if you want a result faster
    # the validation set is used to identify parameter settings (epoch) that achieves
    # the highest classification accuracy
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer="adamax")

    return model
'''

