import feature_extraction as feature
import classifiers as cls


def train(classifiers, train_type, **kwargs):

    for classifier in classifiers:
        classifier.train(train_type, **kwargs)


def test(models, test_type, **kwargs):
    all_preds = []
    for model in models:
        preds, y_test = model.test(test_type, **kwargs)
        all_preds.append(preds)

    labels = [[None, 0] for i in range(len(y_test))]
    for j in range(len(y_test)):
        tot_per, num_est= dict(), dict()
        for i in range(len(all_preds)):
            label, percent = all_preds[i][j]
            try:
                tot_per[label] += percent
            except KeyError:
                tot_per[label] = percent
            try:
                num_est[label] += 1
            except KeyError:
                num_est[label] = 1
        for label in tot_per.keys():
            p = tot_per[label]/num_est[label]
            if p > labels[j][1]:
                labels[j][0] = label
                labels[j][1] = p
    labels = cls.np.array(labels).astype(cls.np.int8)
    return labels, y_test


def preprocessing(train_batch, test_batch, train_subsetn):

    x_train, y_train, x_test, y_test = feature.read_instructions('dataset.txt')

    if train_subsetn != 'all':
        train_dataset = feature.data_set(x_train, y_train)
        x_train, y_train = feature.subsetn_random(train_dataset, train_subsetn)
    feature.extract_features(x_train, y_train, batch=train_batch, extract_type='train')

    feature.extract_features(x_test, y_test, batch=test_batch, extract_type='test')


def main(classifiers,train_batch, test_batch, train_type='batch', preprocess_data=True, train_model=True, test_data=True,
         train_subsetn=50, test_type='batch', **kwargs):

    if preprocess_data:
        print()
        preprocessing(train_batch, test_batch, train_subsetn)
    classifiers = cls.get_model(classifiers)

    if train_model:
        print('INFO: Training is starting!')
        train(classifiers, train_type, **kwargs)
        print('INFO: Training is completed!')

    if test_data:
        models = cls.load_models(cls.all_models_path)
        if len(models) == 0:
            print('Any model to load!!')
            exit(0)
        print('INFO: Testing is starting!')
        labels, y_test = test(models, test_type, **kwargs)
        print('Accuracy:,', cls.evaluate_accuracy(labels[:,0], y_test))
        print('INFO: Testing is completed!')

        return labels, y_test
"""
from version .1
def test(x_test, y_test, models, **kwargs):

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

    labels = cls.np.array(labels).astype(cls.np.int8)
    return labels[:, 0]
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
    labels = test(x_test, y_test, models)
    print('Test is completed')

    accuracy = cls.evaluate_accuracy(labels, y_test)
    print('Accuracy is', accuracy)
def train(train_set, classifiers=('cnn', 'cnn'), train_subset=2, fit_time_per_model=1, feature_type='melspectogram',
          **kwargs):

    for classifier in classifiers:
        classifier.train(train_set, train_subset, fit_time_per_model, feature_type, **kwargs)

    # save_all_models(classifiers)
"""


# https://github.com/jaron/deep-listening
ml_classifiers = ['svm']
rt = main(ml_classifiers, train_batch='all', test_batch='all', preprocess_data=False,  train_model=False, train_type='batch', test_type='batch',
     train_subsetn=50, test_data=True)

# ml_classifiers = ['nn']
# nn_layers = [30, 10]
# main(ml_classifiers, 1, epoch=2, batch_size=256, train_model=True, nn_layers=nn_layers)

