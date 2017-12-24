import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

import feature_extraction as feature
import classifiers as cls
import os
import numpy as np
models_path = 'models'


def train(train_set, classifiers=('cnn', 'cnn'), train_subset=2, fit_time_per_model=1, **kwargs):

    models = []

    for classifier in classifiers:
        xt, yt = feature.subsetn_random(train_set, train_subset)
        ft, lt = feature.extract(xt, yt)
        print('creating model', classifier.name)
        nb_classes = len(np.unique(lt))

        x_train, y_train = classifier.prepare_data(ft, lt)
        classifier.build_model(x_train, nb_classes=nb_classes)

        for fit_t in range(fit_time_per_model):
            if fit_t != 0:
                x_train, y_train = classifier.prepare_data(ft, lt)

            classifier.fit(x_train, y_train, **kwargs)
        models.append(classifier)

    save_all_models(models)


def load_models(path):
    models = list()
    for dir_name, subdir_list, file_list in os.walk(path):
        for f_name in file_list:
            file_path = os.path.join(dir_name, f_name)
            model = cls.mappings[dir_name.split('_')[0]]  # create empty model
            model.load(file_path)                           # load model
            models.append(model)
    return models


def save_all_models(models):

    for classifier in models:
        i = 0
        classifier.save(i)
        i += 1


def main(classifiers, train_subset=2, fit_time_per_model=1, use_model=True):

    classifiers = cls.get_model(classifiers)

    x_train, y_train, x_test, y_test = feature.read_instructions('dataset_2.txt')
    train_set = feature.data_set(x_train, y_train)

    print('data preprocessing is completed! Starts to train')
    # test_set = feature.data_set(x_test, y_test)
    train(train_set, classifiers, train_subset, fit_time_per_model)

    if use_model:
        print('Training is completed, loading models')
        models = load_models(models_path)

    print('loading models completed, test is starting ')
    exit(0)
    # test(x_test, y_test, models)
    print('test is completed')


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




ml_classifiers = ['cnn', 'cnn']
main(ml_classifiers, 50)
