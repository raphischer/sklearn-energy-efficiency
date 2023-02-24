import os
import re
import shutil

from evaluate_hyperparams import classifiers

old_dir = 'sklearn_hyperparameters/hyperparameters_2022_10_14_16_44_47'
new_dir = 'sklearn_hyperparameters/hyperparameters_2022_11_07_16_44_47'

lookup = {
    'Ridge': 'Ridge Regression',
    'Naive Bayes': 'Gaussian Naive Bayes',
    'Nearest Neighbors': 'k-Nearest Neighbors'
}

if not os.path.isdir(new_dir):
    os.makedirs(new_dir)

for fname in sorted(os.listdir(old_dir)):
    match = re.match('hyperparameters__(.*)__(.*).json', fname)
    try:
        ds = match.group(1)
        method = match.group(2).replace('_', ' ')
        if method in lookup:
            method = lookup[method]
        if method in classifiers:
            new_method = method
        else:
            for name, (short, _, _, _) in classifiers.items():
                if short == method:
                    new_method = name
                    break
            else:
                raise ValueError
        new_method = new_method.replace(' ', '_')
        old_file = os.path.join(old_dir, fname)
        new_file = os.path.join(new_dir, f'hyperparameters__{ds}__{new_method}.json')
        print(f'Copying  {fname:<60} to {os.path.basename(new_file):<60}')
        shutil.copyfile(old_file, new_file)
        
    except Exception:
        print(f'Not copy {fname:<60}')
        