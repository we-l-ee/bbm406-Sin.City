import numpy as np
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras import backend as K
from keras import optimizers
from keras.callbacks import EarlyStopping

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn import svm
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

    def estimatep(self, pred):

        labels, counts = pred

        index = np.argmax(counts)
        sum_ = np.sum(counts)
        percent = (counts[index] / sum_) * 100
        label = labels[index]

        return label, percent

    def test(self, test):
        ''''
         process='all-predictions
        def default(predicted_labels):
            return np.unique(predicted_labels, return_counts=True)
    
        all_func = {'all-predictions': default, 'estimate-percent': estimatep}
        func = all_func[process]
        '''
        preds = []
        for path in test:
            f = feature.extract([path])
            x_test, null = self.prepare_data(f, None)
            pred = self.estimatep(np.unique(self.predict(x_test), return_counts=True))
            preds.append(pred)
        return preds


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
            nb_classes = len(np.unique(lt))
            y_train = np_utils.to_categorical(lt, nb_classes)
        return x_train, y_train

    def fit(self, x_train, y_train, **kwargs):
        self.model.fit(x_train, y_train,
                       batch_size=kwargs['batch_size'],
                       epochs=kwargs['epoch'], validation_split=0.1)

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
            nb_classes = len(np.unique(lt))
            y_train = np_utils.to_categorical(lt, nb_classes)
        return x_train, y_train

    def fit(self, x_train, y_train, **kwargs):
        self.model.fit(x_train, y_train,
                       batch_size=kwargs['batch_size'],
                       epochs=kwargs['epoch'], validation_split=0.1)

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

    def predict(self, x_test, **kwargs):
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = 32
        preds = self.model.predict(x_test, batch_size=kwargs['batch_size'], verbose=1)
        return [np.argmax(pred) for pred in preds]


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
        model = svm.LinearSVC()
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

def build_model():
    frames = 41
    bands = 60
    feature_size = bands * frames  # 60x41
    num_labels = 10
    num_channels = 2

    # input: 60x41 data frames with 2 channels => (60,41,2) tensors

    # filters of size 1x1
    f_size = 1

    # first layer has 48 convolution filters
    model = Sequential()
    model.add(Conv2D(48, f_size, strides=f_size, kernel_initializer='normal', padding='same',
                            input_shape=(bands, frames, num_channels)))
    model.add(Conv2D(48, f_size, strides=f_size, kernel_initializer='normal', padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # next layer has 96 convolution filters
    model.add(Conv2D(96, f_size, strides=f_size, kernel_initializer='normal', padding='same'))
    model.add(Conv2D(96, f_size, strides=f_size, kernel_initializer='normal', padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # flatten output into a single dimension
    # Keras will do shape inference automatically
    model.add(Flatten())

    # then a fully connected NN layer
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # finally, an output layer with one node per class
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    # use the Adam optimiser
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    return model


def evaluate(model, x_test, y_test):
    y_prob = model.predict_proba(x_test, verbose=0)
    y_pred = y_prob.argmax(axis=-1)
    y_true = np.argmax(y_test, 1)

    roc = roc_auc_score(y_test, y_prob)
    print("ROC:", round(roc, 3))

    # evaluate the model
    score, accuracy = model.evaluate(x_test, y_test, batch_size=32)
    print("\nAccuracy = {:.2f}".format(accuracy))

    # the F-score gives a similiar value to the accuracy score, but useful for cross-checking
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='micro')
    print("F-Score:", round(f, 2))

    return roc, accuracy



def train(model,x_train ,y_train):
    data_dir = "data/us8k-np-cnn"
    all_folds = False
    av_acc = 0.
    av_roc = 0.
    num_folds = 0

    # as we use two folds for training, there are 9 possible trails rather than 10
    max_trials = 5

    # earlystopping ends training when the validation loss stops improving
    earlystop = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')

    if all_folds:
        feature.load_all_folds()
        model.fit(x_train, y_train , callbacks=[earlystop], batch_size=32, epochs=1)
    else:
        # use folds incrementally
        for f in range(1, max_trials + 1):
            num_folds += 1
            v = f + 2
            if v > 10: v = 1
            t = v + 1
            if t > 10: t = 1

            print("\n*** Train on", f, "&", (f + 1), "Validate on", v, "Test on", t, "***")

            # load two folds for training data
            train_x, train_y = feature.load_folds([f, f + 1])

            # load one fold for validation
            valid_x, valid_y = feature.load_folds([v])

            # load one fold for testing
            test_x, test_y = feature.load_folds([t])

            print("Building model...")
            model = build_model()

            # now fit the model to the training data, evaluating loss against the validation data
            print("Training model...")
            model.fit(train_x, train_y, validation_data=(valid_x, valid_y), callbacks=[earlystop], batch_size=64,
                      epochs=1)

            # now evaluate the trained model against the unseen test data
            print("Evaluating model...")
            roc, acc = evaluate(model)
            av_roc += roc
            av_acc += acc

    print('\nAverage R.O.C:', round(av_roc / max_trials, 3))
    print('Average Accuracy:', round(av_acc / max_trials, 3))


def prediction(sound_names, model,parent_dir ,sound_file_paths):
    for s in range(len(sound_names)):

        print("\n----- ", sound_names[s], "-----")
        # load audio file and extract features
        predict_file = parent_dir + sound_file_paths[s]
        predict_x = feature.extract_feature_array(predict_file)

        # generate prediction, passing in just a single row of features
        predictions = model.predict(predict_x)

        if len(predictions) == 0:
            print("No prediction")
            continue

        # for i in range(len(predictions[0])):
        #    print sound_names[i], "=", round(predictions[0,i] * 100, 1)

        # get the indices of the top 2 predictions, invert into descending order
        ind = np.argpartition(predictions[0], -2)[-2:]
        ind[np.argsort(predictions[0][ind])]
        ind = ind[::-1]

        print("Top guess: ", sound_names[ind[0]], " (", round(predictions[0, ind[0]], 3), ")")
        print("2nd guess: ", sound_names[ind[1]], " (", round(predictions[0, ind[1]], 3), ")")