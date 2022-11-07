from multiprocessing.sharedctypes import Value
from shutil import ExecError
import time
import os
import json
from typing import Type

import numpy as np
from scipy.stats import uniform
from scipy import sparse

# sklearn data imports
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer

# sklearn classifier imports
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from mlee.util import create_output_dir, PatchedJSONEncoder

# TOY DATA
# reg_datasets = {ds: getattr(datasets, f'load_{ds}') for ds in ['diabetes']}
# sel_datasets = {ds: getattr(datasets, f'load_{ds}') for ds in ['iris']}#, 'digits', 'wine', 'breast_cancer']}

# REAL WORLD DATA
sel_datasets = [
    # Sklearn real-life datasets
    # 'olivetti_faces',
    # 'lfw_people',
    # 'lfw_pairs',
    # '20newsgroups_vectorized',
    # 'covtype',
    # 'kddcup99',
    # 'rcv1', # TODO SPARSE PROBLEMS ValueError: sparse multilabel-indicator for y is not supported.

    # Popular OpenML datasets
    # 'credit-g',
    'mnist_784',
    # 'SpeedDating',
    # 'phoneme',
    # 'blood-transfusion-service-center'
]

classifiers = {

    "k-Nearest Neighbors": (
        'kNN',
        KNeighborsClassifier(algorithm='auto'),
        {
            'n_neighbors': [1, 3, 5, 10, 15, 20, 30, 50],
            'algorithm': ['auto', 'ball_tree', 'kd_tree'],
            'leaf_size': [10, 20, 30],
            'p': [1, 2, 3]

        },
        lambda clf: clf.n_features_in_ * clf.n_samples_fit_ 
    ),
    
    "Support Vector Machine": (
        'SVM',
        SVC(), 
        {
            'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
            'C': uniform(0, 2),
            'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1]
        },
        lambda clf: sum([clf.class_weight_.size, clf.intercept_.size, clf.support_vectors_.size])
    ),

    "Random Forest": (
        'RF',
        RandomForestClassifier(), 
        {
            "n_estimators": [10, 20, 40, 75, 100, 150],
            "criterion": ["gini", "entropy"],
            "max_depth": [10, 5],
            "max_features": ['sqrt', 'log2', 5, 10, 20],
        },
        # n_params = 2 * number of nodes (feature & threshold)
        lambda clf: sum([tree.tree_.node_count * 2 for tree in clf.estimators_])
    ),

    "Extra Random Forest": (
        'XRF',
        ExtraTreesClassifier(), 
        {
            "n_estimators": [10, 20, 40, 75, 100, 150],
            "criterion": ["gini", "entropy"],
            "max_depth": [10, 5],
            "max_features": ['sqrt', 'log2', 5, 10, 20],
        },
        # n_params = 2 * number of nodes (feature & threshold)
        lambda clf: sum([tree.tree_.node_count * 2 for tree in clf.estimators_])
    ),

    "AdaBoost": (
        'AB',
        AdaBoostClassifier(),
        {
            "n_estimators": [10, 20, 40, 75, 100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            "algorithm": ['SAMME', 'SAMME.R']
        },
        # n_params = 2 * number of nodes (feature & threshold)
        lambda clf: sum([tree.tree_.node_count * 2 for tree in clf.estimators_])
    ),

    "Gaussian Naive Bayes": (
        'GNB',
        GaussianNB(),
        {
            "var_smoothing": [1e-6, 1e-9, 1e-12]
        },
        lambda clf: sum([clf.class_prior_.size, clf.epsilon_, ])
    ),

    "Ridge Regression": (
        'RR',
        linear_model.RidgeClassifier(),
        {
            'alpha': uniform(0, 2)
        },
        lambda clf: sum([clf.coef_.size, clf.intercept_.size])
    ),

    "Logistic Regression": (
        'LR',
        linear_model.LogisticRegression(max_iter=500),
        {
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'C': uniform(0, 2),
            'solver': ['lbfgs', 'sag', 'saga'],
        },
        lambda clf: sum([clf.coef_.size, clf.intercept_.size])
    ),

    "Linear Stochastic Gradient Descent": (
        'SGD',
        linear_model.SGDClassifier(max_iter=500),
        {
            "loss" : ['hinge', 'log_loss', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
            "penalty": ['l2', 'l1', 'elasticnet'],
            'alpha': uniform(0, 2)
        },
        lambda clf: sum([clf.coef_.size, clf.intercept_.size])
    ),

    "Multilayer Perceptron": (
        'MLP',
        MLPClassifier(max_iter=500, early_stopping=True),
        {
            "hidden_layer_sizes" : [ (200,), (100,), (50,), (100, 50,), (80, 50,) ],
            "solver": ['sgd', 'adam'],
            "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1],
            "learning_rate_init": [0.00001, 0.0001, 0.001, 0.01, 0.1],
        },
        lambda clf: sum([layer_w.size for layer_w in clf.coefs_] + [layer_i.size for layer_i in clf.intercepts_])
    )

    # "Gauss Process": (
    #     GaussianProcessClassifier(),
    #     {
    #         "kernel": [
    #             kernels.Matern(length_scale=1.0, nu=0.5),
    #             kernels.Matern(length_scale=1.0, nu=1.5),
    #             kernels.Matern(length_scale=1.0, nu=2.5),
    #             kernels.Matern(length_scale=0.5, nu=0.5),
    #             kernels.Matern(length_scale=0.5, nu=1.5),
    #             kernels.Matern(length_scale=0.5, nu=2.5),
    #             kernels.Matern(length_scale=2.0, nu=0.5),
    #             kernels.Matern(length_scale=2.0, nu=1.5),
    #             kernels.Matern(length_scale=2.0, nu=2.5),
    #             kernels.RBF(length_scale=1.0),
    #             kernels.RBF(length_scale=0.5),
    #             kernels.RBF(length_scale=2.0),
    #             kernels.ConstantKernel(constant_value=1.0),
    #             kernels.ConstantKernel(constant_value=0.5),
    #             kernels.ConstantKernel(constant_value=2.0),
    #             kernels.RationalQuadratic(length_scale=1.0, alpha=1.0),
    #             kernels.RationalQuadratic(length_scale=1.0, alpha=0.5),
    #             kernels.RationalQuadratic(length_scale=1.0, alpha=2.0),
    #             kernels.RationalQuadratic(length_scale=0.5, alpha=1.0),
    #             kernels.RationalQuadratic(length_scale=0.5, alpha=0.5),
    #             kernels.RationalQuadratic(length_scale=0.5, alpha=2.0),
    #             kernels.RationalQuadratic(length_scale=2.0, alpha=1.0),
    #             kernels.RationalQuadratic(length_scale=2.0, alpha=0.5),
    #             kernels.RationalQuadratic(length_scale=2.0, alpha=2.0),
    #             kernels.ExpSineSquared(length_scale=1.0, periodicity=1.0),
    #             kernels.ExpSineSquared(length_scale=1.0, periodicity=0.5),
    #             kernels.ExpSineSquared(length_scale=1.0, periodicity=2.0),
    #             kernels.ExpSineSquared(length_scale=0.5, periodicity=1.0),
    #             kernels.ExpSineSquared(length_scale=0.5, periodicity=0.5),
    #             kernels.ExpSineSquared(length_scale=0.5, periodicity=2.0),
    #             kernels.ExpSineSquared(length_scale=2.0, periodicity=1.0),
    #             kernels.ExpSineSquared(length_scale=2.0, periodicity=0.5),
    #             kernels.ExpSineSquared(length_scale=2.0, periodicity=2.0),
    #         ],
    #         'n_restarts_optimizer': [0, 1, 2, 3, 4, 5]
    #     },
    #     lambda clf: 0
    # )
}



################## TODO ##################
# 20newsgroups + NB
# covtype SVM
# lfw people logreg
# mnist logreg

def label_encoding(X_train, X_test=None):
    old_shape = X_train.shape
    if X_test is None:
        data = X_train
    else:
        data = np.concatenate([X_train, X_test])
    categorical = []
    if len(data.shape) > 1:
        for column in range(data.shape[1]):
            try: 
                float_col = data[:, column].astype(float)
                data[:, column] = float_col
            except Exception:
                categorical.append(column)
                data[:, column] = preprocessing.LabelEncoder().fit_transform(data[:, column])
    else:
        data = preprocessing.LabelEncoder().fit_transform(data)
    if X_test is None:
        X_train = data
    else:
        X_train, X_test = np.split(data, [X_train.shape[0]])
    assert(X_train.shape == old_shape)
    return X_train, X_test, categorical


def load_data(ds_name):
    if hasattr(datasets, f'fetch_{ds_name}'):
        ds_loader = getattr(datasets, f'fetch_{ds_name}')
    elif hasattr(datasets, f'load_{ds_name}'):
        ds_loader = getattr(datasets, f'load_{ds_name}')
    else:
        try:
            ds_loader = lambda : datasets.fetch_openml(name=ds_name, as_frame=False)
        except:
            raise RuntimeError(f'{ds_name} data could not be found!')
    try:
        # some datasets come with prepared split
        ds_train = ds_loader(subset='train')
        X_train = ds_train.data
        y_train = ds_train.target
        ds_test = ds_loader(subset='test')
        X_test = ds_test.data
        y_test = ds_test.target
        if X_train.shape == X_test.shape:
            raise TypeError # some data sets allow for specific subsets, but return the full dataset if subset is not selected well
    except TypeError:
        ds = ds_loader()
        X_train, X_test, y_train, y_test = train_test_split(ds.data, ds.target) # TODO use stratify?
    # remove labels & rows that are only present in one split
    train_labels = set(list(y_train))
    test_labels = set(list(y_test))
    for label in train_labels:
        if label not in test_labels:
            where = np.where(y_train != label)[0]
            X_train, y_train = X_train[where], y_train[where]
    for label in test_labels:
        if label not in train_labels:
            where = np.where(y_test != label)[0]
            X_test, y_test = X_test[where], y_test[where]
    # use label encoding for categorical features and labels
    try:
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        categorical_columns = []
    except ValueError:
        X_train, X_test, categorical_columns = label_encoding(X_train, X_test)
    try:
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
    except ValueError:
        y_train, y_test, _ = label_encoding(y_train, y_test)
    # impute nan values
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    X_train = imp.fit_transform(X_train)
    X_test = imp.fit_transform(X_test)
    # identify the unique categorical values of each column
    cat_vals = [np.array(sorted(set(np.concatenate([np.unique(X_train[:, col]), np.unique(X_test[:, col])])))) for col in categorical_columns]
    # onehot encoding for categorical features, standard-scale all non-categoricals
    if not sparse.issparse(X_train):
        scaler = ColumnTransformer([
            ('categorical', preprocessing.OneHotEncoder(categories=cat_vals), categorical_columns)
        ], remainder=preprocessing.StandardScaler())
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    dir = create_output_dir(dir='sklearn_hyperparameters', prefix='hyperparameters')
    n_jobs = 10
    
    for ds_name in sel_datasets:
        X_train, X_test, y_train, y_test = load_data(ds_name)

        # #### TEST DATASET
        # clf = GaussianProcessClassifier()
        # clf.fit(X_train, y_train)
        # for split, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
        #     pred = clf.predict(X)
        #     print(f'{ds_name:<25} {str(X_train.shape):<13} {split:<6} accuracy {accuracy_score(y, pred)*100:6.2f}')




        #### RANDOMSEARCH
        try:
            for name, (_, classifier, cls_params, _) in classifiers.items():
                print(f'Running hyperparameter search for {ds_name:<15} {name:<18}')
                # t_start = time.time()
                multithread_classifier = 'n_jobs' in classifier.get_params().keys()
                if multithread_classifier:
                    classifier.set_params(**{'n_jobs': n_jobs})
                    clf = RandomizedSearchCV(classifier, cls_params, random_state=0, n_iter=50, verbose=6, n_jobs=None)
                else:
                    clf = RandomizedSearchCV(classifier, cls_params, random_state=0, n_iter=50, verbose=6, n_jobs=n_jobs)
                search = clf.fit(X_train, y_train)
                with open(os.path.join(dir, f'hyperparameters__{ds_name}__{name.replace(" ", "_")}.json'), 'w') as outfile:
                    json.dump(clf.cv_results_, outfile, indent=4, cls=PatchedJSONEncoder)
        except Exception as e:
            print(e)



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
