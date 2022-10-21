import time
import os
import json

from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from scipy.stats import uniform

from mlee.util import create_output_dir, PatchedJSONEncoder

# TOY DATA
# reg_datasets = {ds: getattr(datasets, f'load_{ds}') for ds in ['diabetes']}
# sel_datasets = {ds: getattr(datasets, f'load_{ds}') for ds in ['iris']}#, 'digits', 'wine', 'breast_cancer']}

# REAL WORLD DATA
sel_datasets = [
    # 'olivetti_faces',
    'lfw_people',
    '20newsgroups_vectorized',
    'covtype',
    # 'rcv1', # SPARSE PROBLEMS ValueError: sparse multilabel-indicator for y is not supported.
    # 'kddcup99' # FILTER CATEGORICAL COLUMNS ValueError: could not convert string to float: b'tcp'
]

classifiers = {
    # "Nearest Neighbors": (KNeighborsClassifier(algorithm='auto'), {'n_neighbors': np.arange(1, 20)}),
    # "SVM": (SVC(), {
    #     'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
    #     'C': uniform(0, 10),
    #     'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1]
    # }),

    "Random Forest": (
        RandomForestClassifier(), 
        {
            "n_estimators": [10, 20, 40, 75, 100, 150, 200],
            "criterion": ["gini", "entropy"],
            "max_depth": [50, 30, 20, 15, 10, 5],
            "max_features": ['sqrt', 'log2', 5, 10],
        },
        # n_params = 2 * number of nodes (feature & threshold)
        lambda clf: sum([tree.tree_.node_count * 2 for tree in clf.estimators_])
    ),

    "Extra Random Forest": (
        ExtraTreesClassifier(), 
        {
            "n_estimators": [10, 20, 40, 75, 100, 150, 200],
            "criterion": ["gini", "entropy"],
            "max_depth": [50, 30, 20, 15, 10, 5],
            "max_features": ['sqrt', 'log2', 5, 10],
        },
        # n_params = 2 * number of nodes (feature & threshold)
        lambda clf: sum([tree.tree_.node_count * 2 for tree in clf.estimators_])
    ),

    # "AdaBoost": (
    #     AdaBoostClassifier(),
    #     {
    #         "n_estimators": [10, 20, 40, 75, 100, 150, 200],
    #         'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
    #         "algorithm": ['SAMME', 'SAMME.R']
    #     },
    #     # n_params = 2 * number of nodes (feature & threshold)
    #     lambda clf: sum([tree.tree_.node_count * 2 for tree in clf.estimators_])
    # ),

    # "Naive Bayes": (
    #     GaussianNB(),
    #     {
    #         "var_smoothing": [1e-6, 1e-9, 1e-12]
    #     },
    #     lambda clf: sum([clf.class_prior_.size, clf.epsilon_, ])
    # ),

    # "Ridge": (
    #     linear_model.RidgeClassifier(),
    #     {
    #         'alpha': uniform(0, 2)
    #     },
    #     lambda clf: 0
    # ),

    # "Logistic Regression": (
    #     linear_model.LogisticRegression(max_iter=500),
    #     {
    #         'penalty': ['l1', 'l2', 'elasticnet', None],
    #         'C': uniform(0, 10),
    #         'solver': ['lbfgs', 'sag', 'saga'],
    #     },
    #     lambda clf: 0
    # ),

    # "SGD": (
    #     linear_model.SGDClassifier(max_iter=500),
    #     {
    #         "loss" : ['hinge', 'log_loss', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
    #         "penalty": ['l2', 'l1', 'elasticnet'],
    #         'alpha': uniform(0, 2)
    #     },
    #     lambda clf: 0
    # )
}


def label_encoding(data):
    number = preprocessing.LabelEncoder()
    data = number.fit_transform(data)
    return data


def load_data(ds_name):
    if hasattr(datasets, f'fetch_{ds_name}'):
        ds_loader = getattr(datasets, f'fetch_{ds_name}')
    elif hasattr(datasets, f'load_{ds_name}'):
        ds_loader = getattr(datasets, f'load_{ds_name}')
    else:
        raise RuntimeError(f'{ds_name} data could not be found!')
    try:
        # some datasets come with prepared split
        ds_train = ds_loader(subset='train')
        X_train = ds_train.data
        y_train = ds_train.target
        ds_test = ds_loader(subset='test')
        X_test = ds_test.data
        y_test = ds_test.target
    except TypeError:
        ds = ds_loader()
        X_train, X_test, y_train, y_test = train_test_split(ds.data, ds.target)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    dir = create_output_dir(dir='sklearn_hyperparameters', prefix='hyperparameters')
    n_jobs = 10
    
    for ds_name in sel_datasets:
        X_train, X_test, y_train, y_test = load_data(ds_name)

        # #### TEST DATASET
        # clf = RandomForestClassifier(n_estimators=200, max_depth=50, max_features='sqrt', n_jobs=n_jobs)
        # clf.fit(X_train, y_train)
        # for split, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
        #     pred = clf.predict(X)
        #     print(f'KNN on {ds_name:<25} {str(X_train.shape):<13} {split:<6} accuracy {accuracy_score(y, pred)*100:6.2f}')




        #### RANDOMSEARCH
        try:
            for name, (classifier, cls_params, _) in classifiers.items():
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
