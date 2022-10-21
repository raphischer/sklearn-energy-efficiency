import argparse
from datetime import timedelta
import json
import os
import pickle
import time
import sys

import numpy as np
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score, precision_score, recall_score

from mlee.util import fix_seed, create_output_dir, Logger, PatchedJSONEncoder
from mlee.monitoring import Monitoring, monitor_flops_papi
from evaluate_hyperparams import classifiers, load_data


classification_metrics = {
    'accuracy': accuracy_score,
    'f1': lambda y1, y2: f1_score(y1, y2, average='micro'),
    'precision': lambda y1, y2: precision_score(y1, y2, average='micro'),
    'recall': lambda y1, y2: recall_score(y1, y2, average='micro'),
}


# additional metrics: generalization? hyperparameter fitting effort? model size?


def init_model_and_data(args):
    X_train, X_test, y_train, y_test = load_data(args.dataset)
    clf = classifiers[args.model][0]   
    fname = os.path.join(args.hyperparameters, f'hyperparameters__{args.dataset}__{args.model.replace(" ", "_")}.json')
    with open(fname, 'r') as hyperf:
        hyper_content = json.load(hyperf)
    best_rank = hyper_content['rank_test_score'].index(1)
    best_params = hyper_content['params'][best_rank]
    clf.set_params(**best_params)
    return X_train, X_test, y_train, y_test, clf


def finalize_model(clf, output_dir, n_params):
    model_fname = os.path.join(output_dir, 'model.pkl')
    with open(model_fname, 'wb') as modelfile:
        pickle.dump(clf, modelfile)    

    # count flops of infering single random data row
    test_data = np.random.rand(1, clf.n_features_in_)
    flops = monitor_flops_papi(lambda : clf.predict(test_data))

    clf_info = {
        'hyperparams': clf.get_params(),
        'params': n_params,
        'fsize': os.path.getsize(model_fname),
        'flops': flops
    }
    return clf_info


def evaluate_single(args):
    t0 = time.time()
    args.seed = fix_seed(args.seed)

    step = 'train' ############## TRAINING ##############
    output_dir = create_output_dir(args.output_dir, step, args.__dict__)
    tmp = sys.stdout # reroute the stdout to logfile, remember to call close!
    sys.stdout = Logger(os.path.join(output_dir, f'logfile.txt'))

    X_train, X_test, y_train, y_test, clf = init_model_and_data(args)
    monitoring = Monitoring(args.gpu_monitor_interval, args.cpu_monitor_interval, output_dir, f'{step}_')
    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    monitoring.stop() 
    model_info = finalize_model(clf, output_dir, classifiers[args.model][2](clf))

    results = {
        'history': {}, # TODO track history
        'start': start_time,
        'end': end_time,
        'model': model_info
    }
    # write results
    with open(os.path.join(output_dir, f'{step}_results.json'), 'w') as rf:
        json.dump(results, rf, indent=4, cls=PatchedJSONEncoder)



    step = 'infer' ############## INFERENCE ##############
    output_dir = create_output_dir(args.output_dir, step, args.__dict__)
    monitoring = Monitoring(args.gpu_monitor_interval, args.cpu_monitor_interval, output_dir, f'{step}_')
    start_time = time.time()
    y_pred = clf.predict(X_test)
    end_time = time.time()
    monitoring.stop()

    results = {
        'metrics': {},
        'start': start_time,
        'end': end_time,
        'model': model_info
    }
    # calculating predictive quality metrics
    for score, func in classification_metrics.items():
        try: # some score metrics need information on available classes
            results['metrics'][score] = func(y_test, y_pred, labels=clf.classes_)
        except TypeError:
            results['metrics'][score] = func(y_test, y_pred)
    if hasattr(clf, 'predict_proba'):
        y_proba = clf.predict_proba(X_test)
        results['metrics']['top_5'] = top_k_accuracy_score(y_test, y_proba, k=5, labels=clf.classes_)
    # write results
    with open(os.path.join(output_dir, f'{step}_results.json'), 'w') as rf:
        json.dump(results, rf, indent=4, cls=PatchedJSONEncoder)



    print(f"Evaluation finished in {timedelta(seconds=int(time.time() - t0))} seconds, results can be found in {output_dir}")
    sys.stdout.close()
    sys.stdout = tmp
    return output_dir


def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description="Classification training with Tensorflow, based on PyTorch training", add_help=add_help)

    # data and model input
    parser.add_argument("--dataset", default="olivetti_faces")
    parser.add_argument("--model", default="Random Forest")
    parser.add_argument("--hyperparameters", default="sklearn_hyperparameters/hyperparameters_2022_10_14_16_44_47")
    # output
    parser.add_argument("--output-dir", default='logs/sklearn', type=str, help="path to save outputs")
    parser.add_argument("--gpu-monitor-interval", default=.01, type=float, help="Setting to > 0 activates GPU profiling every X seconds")
    parser.add_argument("--cpu-monitor-interval", default=.01, type=float, help="Setting to > 0 activates CPU profiling every X seconds")

    # randomization and hardware
    parser.add_argument("--seed", type=int, default=42, help="Seed to use (if -1, uses and logs random seed)")
    
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if args.model == 'all':
        for clf in classifiers.keys():
            args.model = clf
            evaluate_single(args)
    else:
        evaluate_single(args)