import argparse
import os

import feature_extraction as feature
import keras
import librosa as lb

import numpy as np
import tflearn
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

path = "D:/wd/bbm406/dataset_s"
# @res_type=scipy sucks
# y,sr = lb.core.load(path)


def init_nn(in_dim, units, activations):
    model = keras.models.Sequential()
    model.add(Dense(units=units[0], activation=[0], input_dim=in_dim))
    for u, a in zip(units[1:], activations[1:]):
        model.add(Dense(units=u, activation=a))
	model.compile(loss='categorical_crossentropy',
		  optimizer='sgd',
		  metrics=['accuracy'])
    return model
""""
x, y = feature.read_instructions()
train_set = feature.data_set(x, y)
xx, yy = feature.subsetn_random(train_set)
f, l = feature.extract(xx, yy)


model = init_nn(1292, [30,2], ['relu','relu','softmax'])

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

"""

""""
def extract_feature(dataset='datasets', stride=128):
    features, labels = [], []
    for root, sub, flist in os.walk(dataset):
        for f in flist:
            path = os.path.join(root, f)
            print(path)
            y, sr = lb.core.load(path,offset=30, duration=30)
            features.append(lb.feature.mfcc(y, sr, n_mfcc=10))
            labels.append(f.split("_")[0], )




    return features, labels
def main():

    parser = argparse.ArgumentParser(description='.')
    parser.add_argument("-data_path", default='',
                        help="Folder to the train data and validation data. It will look for 'train-data.npy'"
                             "and 'validation-data.npy' under this folder unless specified by arguments. "
                             "Default current path(./)"
                             "This is only the path so careful given paths as 'foo/' as meaning under "
                             "the search the validation and train files under the directory of foo")
    parser.add_argument("-model_path", default='',
                        help="NN model that can be used for training or testing. "
                             "Default current path(./). "
                             "This is only the path so careful given paths as 'foo/' as meaning under "
                             "the search the validation and train files under the directory of foo")

    #parser.add_argument("-sigmoid", dest='activation', action='store_const', const=[sigmoid, dsigmoid],
    #                    default=[sigmoid, dsigmoid], help="To use sigmoid as activation function. Default (sigmoid)")
    #parser.add_argument("-tanh", dest='activation', action='store_const', const=[tanh, dtanh]
    #                    , help="To use tanh as activation function. Default (sigmoid)")
    #parser.add_argument("-relu", dest='activation', action='store_const', const=[relu, drelu]
    #                    , help="To use relu as activation function. Default (sigmoid)")

    parser.add_argument("-epsilon", type=float, default=0.003,
                        help="Epsilon value that will be used in gradient ascent as steps length. Default (0.003)")
    parser.add_argument("-lamb", type=float, default=0.01,
                        help="Lambda value that will be used in regularization as factor. (0.01)")
    parser.add_argument("-epocs", type=int, default=50,
                        help="Epocs of the training. Default (10)")
    parser.add_argument("-batch", type=int, default=200,
                        help="Batch size of the training. Default (200)")

    parser.add_argument("-validate", action='store_true',
                        help="Enables validation after training. Data path folder and file name must be "
                             "specified if default values will not be used, see -ndvalidation and -nlvalidation. "
                             "Default (False)")
    parser.add_argument("-onlyvalidate", action='store_true',
                        help="When this is used it will behave as -usemodel activated. It will load a model."
                             " It will not try to learn anything it only validate and show the result.")

    parser.add_argument("-hdims", type=int, nargs='+', default=[120],
                        help="Hidden dimensions of nn. It takse list of integer values "
                             "for each layer and for every layer the integer value"
                             "is that layers size. Default ([120])")
    parser.add_argument("-dsave", dest='save', action='store_false', help="Disables saving after nn model trained. "
                                                                          "Default (True)")
    parser.add_argument("-usemodel", action='store_true', help="Loads already initialize nn model and trains that model "
                        "instead of initializing from scratch. Default (False)")

    parser.add_argument("-nlmodel", default='nn-model.npy', help="Name of the model file that will be load."
                        "Default (nn-model.npy)")
    parser.add_argument("-nsmodel", default='nn-model', help="Name of the model file that will be saved. "
                        "Default (nn-model), it will be saved nn-model.npy thus give this argument as only name.")

    parser.add_argument("-ndtrain", default='train-data.npy', help="Name of the train data file. "
                        "Default (train-data.npy)")
    parser.add_argument("-ndvalidation", default='validation-data.npy', help="Name of the validation data file. "
                        "Default (validation-data.npy)")
    parser.add_argument("-nltrain", default='train-label.npy', help="Name of the train label file. "
                        "Default (train-label.npy)")
    parser.add_argument("-nlvalidation", default='validation-label.npy', help="Name of the validation label file. "
                        "Default (train-label.npy")

    args = parser.parse_args(ARGV[1:])

    dtrain, ltrain = args.data_path + args.ndtrain, args.data_path + args.nltrain
    if not args.onlyvalidate:
        train_data = normalize_data(np.load(dtrain))
        train_label = np.load(ltrain)
        labels = np.unique(train_label)

    dvalidation, lvalidation = args.data_path + args.ndvalidation, args.data_path + args.nlvalidation
    if args.validate or args.onlyvalidate:
        validation_data = normalize_data(np.load(dvalidation))
        validation_label = np.load(lvalidation)

    if not args.onlyvalidate:
        dims=[len(train_data[0])]   #   input layer dimensionality
        dims.extend(args.hdims)
        dims.append(len(labels))    #   output layer dimensionality.

    if args.usemodel or args.onlyvalidate:
        nn = load_nn(args.data_path + args.nlmodel)
        dims = [i.shape[0] for i in nn['W']]
        dims.append(nn['W'][-1].shape[1])
        print("Neural Network Model loaded from disk.")
    else:
        nn = init_nn(dims)
        print("New Neural Network Model initialized.")

    print("Train:", not args.onlyvalidate)
    if not args.onlyvalidate: print("Train labels:",labels)
    if not args.onlyvalidate: print("Train data file:", dtrain)
    if not args.onlyvalidate: print("Train label file:", ltrain)
    print("Use validation:", args.validate or args.onlyvalidate)
    if args.validate or args.onlyvalidate: print("Validation data file:", dvalidation)
    if args.validate or args.onlyvalidate: print("Validation label file:", lvalidation)
    print("Use model:", args.usemodel or args.onlyvalidate)
    if args.usemodel or args.onlyvalidate: print("Model load location:",args.data_path + args.nlmodel)
    print("Save model:", args.save and not args.onlyvalidate)
    if args.save and not args.onlyvalidate: print("Model save location:",args.model_path + args.nsmodel + '.npy')
    print("Layers of Neural Network:", len(dims), "(First one input, last one output, others that are in between are hidden layers)")
    print("Sizes of layers:", dims)
    print("Activation function and it`s derivative function:", args.activation)
    if not args.onlyvalidate: print("Epsilon:", args.epsilon)
    if not args.onlyvalidate: print("Lambda:", args.lamb)
    if not args.onlyvalidate: print("Epocs [",args.epocs,"] and batches [",args.batch,"]")


    if not args.onlyvalidate:
        train_nn(train_data, train_label, nn, args.epocs, args.batch, args.epsilon, args.activation[0], args.activation[1])

    if args.validate or args.onlyvalidate:
        acc = validate_nn(nn, validation_data, validation_label)
        print("Accuracy:", acc)

    if args.save and not args.onlyvalidate:
        save_nn(nn, args.model_path + args.nsmodel)

    return args, nn

"""
# if __name__ == '__main__':
#     main()


def build_model_v2(nb_classes, X):

    convnet = input_data(shape=[None, X.shape[2], X.shape[3], 1], name='input')

    convnet = conv_2d(convnet, 64, 2, activation='elu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 128, 2, activation='elu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 256, 2, activation='elu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 512, 2, activation='elu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)

    convnet = fully_connected(convnet, 1024, activation='elu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, nb_classes, activation='softmax')
    convnet = regression(convnet, optimizer='rmsprop', loss='categorical_crossentropy')

    model = tflearn.DNN(convnet)
    # print("    Model created! ✅")
    return model


def build_model_v1(X, nb_classes, nb_layers=4):
    nb_filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size
    input_shape = (1, X.shape[2], X.shape[3])

    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid', input_shape=input_shape))
    model.add(BatchNormalization(axis=1, mode=2))
    model.add(Activation('relu'))

    for layer in range(nb_layers - 1):
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        model.add(BatchNormalization(axis=1, mode=2))
        model.add(ELU(alpha=1.0))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model


def main2(subset_use_number=5):
    x_train, y_train, x_val, y_val, x_test, y_test = feature.read_instructions()
    train_set = feature.data_set(x_train, y_train)
    val_set = feature.data_set(x_train, y_train)
    test_set = feature.data_set(x_test, y_test)

    train2(train_set, val_set, subset_use_number)

    cnn_models = np.load('cnn_models.npy')
    predict(test_set, cnn_models)


def train2(train_set, val_set, subset_use_number):
    xt, yt = feature.subsetn_random(train_set)
    ft, lt = feature.extract(xt, yt)

    xv, yv = feature.subsetn_random(val_set)
    fv, lv = feature.extract(xv, yv)

    model = build_model_v1(ft, nb_classes=len(lt))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    batch_size = 128
    nb_epoch = 100
    model.fit(ft, lt, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(fv, lv))

    score = model.evaluate(fv, lv, verbose=0)
    print('Val score:', score[0])
    print('Val accuracy:', score[1])

    for _ in range(subset_use_number-1):
        xt, yt = feature.subsetn_random(train_set)
        ft, lt = feature.extract(xt, yt)

        xv, yv = feature.subsetn_random(val_set)
        fv, lv = feature.extract(xv, yv)

        model.fit(ft, lt, batch_size=batch_size, nb_epoch=nb_epoch,
                  verbose=1, validation_data=(fv, lv))

        score = model.evaluate(fv, lv, verbose=0)
        print('Val score:', score[0])
        print('Val accuracy:', score[1])

    model.save('model/cnn.keras')



def main(cnn_number):

    x_train, y_train, x_test, y_test = feature.read_instructions()
    train_set = feature.data_set(x_train, y_train)
    # test_set = feature.data_set(x_test, y_test)

    train(train_set, cnn_number)

    cnn_models = np.load('models/cnn_models.npy')
    predict(x_test, y_test,cnn_models)


def train(train_set, cnn_number):
    cnn_models = list()
    for _ in range(cnn_number):
        xt, yt = feature.subsetn_random(train_set)
        ft, lt = feature.extract(xt, yt)

        model = build_model_v1(ft, nb_classes=len(lt))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])

        cnn_models.append(model)


    np.save('cnn_models.npy', cnn_models)

def estimatep(pred):
    labels, counts = pred

    index = np.argmax(counts)
    sum_ = np.sum(counts)
    percent = (counts[index]/sum_)*100
    label = labels[index]

    return label, percent


def test(x_test, y_test, cnn_models, process='accuracy-percent', **kwargs):
    def default():
        acc, total = 0, 0
        def estimations(models):
            votes = []
            for pred in models:
                label, percent = pred
                if percent > kwargs['percent']
                    pass

    model_estimations = [predictions(x_test, model, "estimate-percent") for model in cnn_models]

    for pred in model_estimations:
        label, percent = pred
        if percent
            pass

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
            each_model.append(func(model.predict(f)))
        preds.append()

    return preds