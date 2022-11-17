# Energy Efficiency Considerations for Popular AI Benchmarks

Code and results for assesing energy efficiency of machine learning tasks on popular benchmark data sets.
The associated research paper is currently under review.

## Installation
All code was executed with Python 3.8, please refer to [requirements](./requirements.txt) for all dependencies.
Depending on how you intend to use this software, only some packages are required.

## Usage
To investigate the results you can use our publicly available [Energy Label Exploration tool](http://167.99.254.41/), so no code needs to be run on your machine (soon to be updated with newest results).

To start [ELEx](elex.py) locally, simply call `python elex.py` and open the given URL in any webbrowser.
Call `python -m mlee.label_generator` to generate an energy label either for a given data set / task / method / environment, or any of the merged logs (provided via command line).
The [results](./paper_results/) (plots and tables) in the paper were generated with the corresponding [script](create_paper_results.py).

New experiments can also be executed, simply run the [evaluation script](evaluate_sklearn.py).
You can pass the chosen method, software backend and more configuration options via command line.
For each experiment a folder is created, which can be [merged](merge_results.py) into more compact `.json` format.
Hyperparameters for all (benchmark X method) combinations are can be found [in this repo](sklearn_hyperparameters) and were identified by a [random search](evaluate_hyperparams.py).
Note that due to monitoring of power draw, we mainly tested on limited hardware architectures and systems (Linux systems with Intel CPUs).

## Previous Work
We already investigated the [efficiency of ImageNet models](https://github.com/raphischer/imagenet-energy-efficiency).

## Road Ahead
We intend to extend and improve our software framework:
- polish the ELEx tool, allow to execute expeirments locally from GUI
- support more implementations, monitoring options, models, metrics, and tasks
- move beyond sustainability and incorporate other important aspects of trustworthiness
- more improvements based on reviewer feedback

## Reference & Term of Use
Please refer to the [license](.LICENSE.md) for terms of use.
If you use this code or the data, please cite our paper and link back to this repository.

Copyright (c) 2022 Raphael Fischer
