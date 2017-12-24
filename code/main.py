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
        print('creating model', classifier)
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

ml_classifiers = ['cnn', 'cnn']
main(ml_classifiers, 3)
