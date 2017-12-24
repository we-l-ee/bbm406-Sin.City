import feature_extraction as feature
import classifiers as cls
import os
import numpy as np

models_path = 'models'


def train(train_set, classifiers=('cnn', 'cnn'), train_subset=2, fit_time_per_model=1, feature_type='melspectogram',
          **kwargs):

    models = []

    for classifier in classifiers:
        xt, yt = feature.subsetn_random(train_set, train_subset)
        ft, lt = feature.extract(xt, yt, feature_type)
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
            model = cls.mappings[dir_name.split('_')[0]]    # create empty model
            model.load(file_path)                           # load model
            models.append(model)
    return models


def save_all_models(models):

    i = 0
    for classifier in models:
        classifier.save(i)
        i += 1


def main(classifiers, train_subset=2, fit_time_per_model=1, feature_type='melspectogram',  use_model=True, **kwargs):

    classifiers = cls.get_model(classifiers)

    x_train, y_train, x_test, y_test = feature.read_instructions('dataset_2.txt')
    train_set = feature.data_set(x_train, y_train)

    print('data preprocessing is completed! Starts to train')
    # test_set = feature.data_set(x_test, y_test)
    train(train_set, classifiers, train_subset, fit_time_per_model, feature_type, **kwargs)

    if use_model:
        print('Training is completed, loading models')
        models = load_models(models_path)

    print('loading models completed, test is starting ')
    exit(0)
    # test(x_test, y_test, models)
    print('test is completed')


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
ml_classifiers = ['cnn', 'cnn']
main(ml_classifiers, 1, epoch=1, batch_size=256)
