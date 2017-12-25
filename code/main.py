import feature_extraction as feature
import classifiers as cls
import os
import numpy as np


models_path = 'models'


def train(train_set, classifiers=('cnn', 'cnn'), train_subset=2, fit_time_per_model=1, feature_type='melspectogram',
          **kwargs):

    for classifier in classifiers:
        classifier.train(train_set, train_subset, fit_time_per_model, feature_type, **kwargs)

    # save_all_models(classifiers)


def load_models(path):
    models = list()
    for dir_name, subdir_list, file_list in os.walk(path):
        for f_name in file_list:
            file_path = os.path.join(dir_name, f_name)
            model = cls.mappings[dir_name.split('_')[0].split('\\')[1]]    # create empty model
            model.load(file_path)                           # load model
            models.append(model)
    return models


def save_all_models(models):

    i = 0
    for classifier in models:
        classifier.save(i)
        i += 1


def main(classifiers, train_subset=2, fit_time_per_model=1, feature_type='melspectogram', save_model=True,
         use_model=True, **kwargs):

    classifiers = cls.get_model(classifiers)

    x_train, y_train, x_test, y_test = feature.read_instructions('dataset_2.txt')
    train_set = feature.data_set(x_train, y_train)

    print('data preprocessing is completed! Starts to train')
    # test_set = feature.data_set(x_test, y_test)
    if save_model:
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

    labels = [[None, 0]] * len(y_test)
    for estimations in model_estimations:
        for i in range(len(estimations)):
            label, percent = estimations[i]
            if labels[i][0] != label:
                if label[i][1] < percent:
                    label[i][1] = percent
            else:
                labels[i][1] += percent

    return labels


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
            x_test, null = model.prepare_data(f, None)
            each_model.append(func(model.predict(x_test)))

        preds.append(each_model)

    return preds
ml_classifiers = ['cnn', 'cnn']
main(ml_classifiers, 1, epoch=1, batch_size=256, save_model=True)
