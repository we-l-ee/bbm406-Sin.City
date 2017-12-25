import feature_extraction as feature
import classifiers as cls
import os


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


def main(classifiers, train_subset=2, fit_time_per_model=1, feature_type='melspectogram', train_model=True,
         use_model=True, **kwargs):

    classifiers = cls.get_model(classifiers)

    x_train, y_train, x_test, y_test = feature.read_instructions('dataset_2.txt')
    train_set = feature.data_set(x_train, y_train)

    print('Data pre-processing is completed! Starts to train')
    # test_set = feature.data_set(x_test, y_test)
    if train_model:
        train(train_set, classifiers, train_subset, fit_time_per_model, feature_type, **kwargs)

    if use_model:
        print('Training is completed, loading models')
        models = load_models(cls.all_models_path)
        if len(models) == 0:
            print('Any model to load!!')
            exit(0)

    print('Loading models completed, test is starting ')
    # exit(0)
    labels = test(x_test[:2], y_test[:2], models)
    print('Test is completed')
    accuracy = cls.evaluate_accuracy(labels, y_test)
    print('Accuracy is', accuracy)


def test(x_test, y_test, models, **kwargs):

    """"
    process = 'accuracy-percent',
    def default():
        acc, total = 0, 0
        def estimations(models):
            votes = []
            for pred in models:
                label, percent = pred
                if percent > kwargs['percent']:
                    pass
    """
    model_estimations = [model.test(x_test) for model in models]

    labels = [[None, 0]] * len(y_test)
    for estimations in model_estimations:
        for i in range(len(estimations)):
            label, percent = estimations[i]
            if labels[i][0] != label:
                if labels[i][1] < percent:
                    labels[i][1] = percent
                    labels[i][0] = label
            else:
                labels[i][1] += percent
    labels = cls.np.array(labels)

    return labels[:, 1]

ml_classifiers = ['cnn', 'cnn']
main(ml_classifiers, 1, epoch=1, batch_size=256, train_model=False)

# ml_classifiers = ['nn']
# nn_layers = [30, 10]
# main(ml_classifiers, 1, epoch=2, batch_size=256, train_model=True, nn_layers=nn_layers)