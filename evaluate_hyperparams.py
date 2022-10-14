import time
import os
import json

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from scipy.stats import uniform

from mlee.util import create_output_dir, PatchedJSONEncoder

# TOY DATA
# reg_datasets = {ds: getattr(datasets, f'load_{ds}') for ds in ['diabetes']}
# sel_datasets = {ds: getattr(datasets, f'load_{ds}') for ds in ['iris']}#, 'digits', 'wine', 'breast_cancer']}

# REAL WORLD DATA
real_classification = [
    'olivetti_faces',
    # '20newsgroups_vectorized',
    # 'lfw_people',
    'covtype',
    # 'rcv1', # ValueError: sparse multilabel-indicator for y is not supported.
    # 'kddcup99' # ValueError: could not convert string to float: b'tcp'
]

sel_datasets = {ds: getattr(datasets, f'fetch_{ds}') for ds in real_classification}

classifiers = {
    "Nearest Neighbors": (KNeighborsClassifier(algorithm='auto'), {'n_neighbors': np.arange(1, 20)}),
    "SVM": (SVC(), {
        'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
        'C': uniform(0, 10),
        'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1]
    }),
    "Random Forest": (RandomForestClassifier(), {
        "n_estimators": [1, 20, 40, 75, 100, 150, 200, 250],
        "criterion": ["gini", "entropy"],
        "max_depth": [20, 10, 5, None],
        "max_features": ['sqrt', 'log2', 5, 10],
    }),
    "Extra Random Forest": (ExtraTreesClassifier(), {
        "n_estimators": [1, 20, 40, 75, 100, 150, 200, 250],
        "criterion": ["gini", "entropy"],
        "max_depth": [20, 10, 5, None],
        "max_features": ['sqrt', 'log2', 5, 10],
    }),
    "AdaBoost": (AdaBoostClassifier(), {
        "n_estimators": [1, 20, 40, 75, 100, 150, 200, 250],
        'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
        "algorithm": ['SAMME', 'SAMME.R']
    }),
    "Naive Bayes": (GaussianNB(), {
        "var_smoothing": [1e-6, 1e-9, 1e-12]
    }),
    "Ridge": (linear_model.RidgeClassifier(), {'alpha': uniform(0, 2)}),
    "Logistic Regression": (linear_model.LogisticRegression(max_iter=1000), {
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'C': uniform(0, 10),
        'solver': ['lbfgs', 'sag', 'saga'],
    }),
    "SGD": (linear_model.SGDClassifier(max_iter=1000), {
        "loss" : ['hinge', 'log_loss', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
        "penalty": ['l2', 'l1', 'elasticnet'],
        'alpha': uniform(0, 2)
    })
}

dir = create_output_dir()

for ds_name, ds_loader in sel_datasets.items():
    try:
        # some datasets come with prepared split
        ds_train = ds_loader(subset='train')
        X_train = ds_train.data
        y_train = ds_train.target
        # ds_test = ds_loader(subset='test')
        # X_test = ds_test.data
        # y_test = ds_test.target
    except TypeError:
        ds = ds_loader()
        X_train, _, y_train, _ = train_test_split(ds.data, ds.target)

    for name, (classifier, cls_params) in classifiers.items():
        print(f'Running hyperparameter search for {ds_name:<15} {name:<18}')
        # t_start = time.time()
        clf = RandomizedSearchCV(classifier, cls_params, random_state=0, n_iter=5)
        search = clf.fit(X_train, y_train)
        with open(os.path.join(dir, f'hyperparameters__{ds_name}__{name.replace(" ", "_")}.json'), 'w') as outfile:
            json.dump(clf.cv_results_, outfile, indent=4, cls=PatchedJSONEncoder)
        # t_train_end = time.time()
        # result_scores = {'fit_time': t_train_end - t_start}
        # result_scores['inf_time'] = 0
        # for split, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
        #     y_pred = classifier.predict(X)
        #     accuracy = accuracy_score(y, y_pred)
        #     result_scores[f'{split}_acc'] = accuracy * 100
        # result_scores['inf_time'] = time.time() - t_train_end
        # print(f'{ds_name:<15} {str(X_train.shape):<13} {str(X_test.shape):<13} {name:<18} ' + ' - '.join([f'{key} {val:6.2f}' for key, val in result_scores.items()]))
    # print('                :::::::::::::             ')
