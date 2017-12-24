import argparse
import os

import feature_extraction as feature

import numpy as np
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import np_utils

cnn_models_path = 'model/cnn_models/'

""""
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


def build_model(X, nb_classes, nb_layers=2):
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

    
def main(cnn_number=4, train_subset=10, fit_time_per_model=1,):

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
    epochs = 1

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
main(1)

