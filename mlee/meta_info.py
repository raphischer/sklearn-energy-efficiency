# TODO outsource information to json format

HARDWARE_NAMES = {
    'NVIDIA A100-SXM4-40GB': 'A100',
    'Quadro RTX 5000': 'RTX 5000',
    'Intel(R) Xeon(R) W-2155 CPU @ 3.30GHz': 'Xeon(R) W-2155',
    'AMD EPYC 7742 64-Core Processor': 'EPYC 7742'
}

IMAGENET_MODEL_INFO = {
    'ResNet50':          {'epochs': 90, 'url': 'https://arxiv.org/abs/1512.03385'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'ResNet101':         {'epochs': 90, 'url': 'https://arxiv.org/abs/1512.03385'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'ResNet152':         {'epochs': 90, 'url': 'https://arxiv.org/abs/1512.03385'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'VGG16':             {'epochs': 90, 'url': 'https://arxiv.org/abs/1409.1556'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'VGG19':             {'epochs': 90, 'url': 'https://arxiv.org/abs/1409.1556'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'EfficientNetB0':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'EfficientNetB1':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'EfficientNetB2':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'EfficientNetB3':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'EfficientNetB4':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'EfficientNetB5':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'EfficientNetB6':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'EfficientNetB7':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'RegNetX400MF':      {'epochs': 100, 'url': 'https://arxiv.org/abs/2003.13678'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'RegNetX32GF':       {'epochs': 100, 'url': 'https://arxiv.org/abs/2003.13678'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'RegNetX8GF':        {'epochs': 100, 'url': 'https://arxiv.org/abs/2003.13678'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'ResNext50':         {'epochs': 100, 'url': 'https://arxiv.org/abs/1611.05431'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'ResNext101':        {'epochs': 100, 'url': 'https://arxiv.org/abs/1611.05431'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'DenseNet121':       {'epochs': 90, 'url': 'https://arxiv.org/pdf/1608.06993'},
    'DenseNet169':       {'epochs': 90, 'url': 'https://arxiv.org/pdf/1608.06993'},
    'DenseNet201':       {'epochs': 90, 'url': 'https://arxiv.org/pdf/1608.06993'},
    'Xception':          {'epochs': None, 'url': 'https://arxiv.org/abs/1610.02357'}, # no information on epochs
    'InceptionResNetV2': {'epochs': 200, 'url': 'https://arxiv.org/abs/1602.07261'},
    'InceptionV3':       {'epochs': 100, 'url': 'https://arxiv.org/abs/1512.00567'},
    'NASNetMobile':      {'epochs': 100, 'url': 'https://arxiv.org/pdf/1707.07012'},
    'MobileNetV2':       {'epochs': 300, 'url': 'https://arxiv.org/abs/1801.04381'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'MobileNetV3Small':  {'epochs': 600, 'url': 'https://arxiv.org/pdf/1905.02244'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'MobileNetV3Large':  {'epochs': 600, 'url': 'https://arxiv.org/pdf/1905.02244'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'QuickNetSmall':     {'epochs': 600, 'url': 'https://arxiv.org/abs/2011.09398'}, # https://github.com/larq/zoo/blob/main/larq_zoo/training/sota_experiments.py
    'QuickNet':          {'epochs': 600, 'url': 'https://arxiv.org/abs/2011.09398'}, # https://github.com/larq/zoo/blob/main/larq_zoo/training/sota_experiments.py
    'QuickNetLarge':     {'epochs': 600, 'url': 'https://arxiv.org/abs/2011.09398'} # https://github.com/larq/zoo/blob/main/larq_zoo/training/sota_experiments.py
}

TASK_TYPES = {
    'infer': 'inference',
    'train': 'training',
}

DATASET_INFO = {
    'olivetti_faces':                   {'name': 'Olivetti faces',       'url': ''},
    '20newsgroups_vectorized':          {'name': '20 Newsgroups',        'url': ''},
    'lfw_people':                       {'name': 'LFW People',           'url': ''},
    'lfw_pairs':                        {'name': 'LFW Pairs',            'url': ''},
    'covtype':                          {'name': 'Covertype',            'url': ''},
    'credit-g':                         {'name': 'German Credit',        'url': ''},
    'mnist_784':                        {'name': 'MNIST',                'url': ''}, 
    'blood-transfusion-service-center': {'name': 'Blood Transf. SC',     'url': 'https://www.openml.org/search?type=data&sort=runs&id=1464&status=active'}
}


def load_model_info(dataset, model_name):
    # TODO update with picking among further info dicts based on dataset
    try:
        info = IMAGENET_MODEL_INFO[model_name]
    except KeyError:
        info = {
            'epochs': None,
            'url': 'https://scikit-learn.org/stable/user_guide.html' # TODO later update with paper URL
        }
    return info


def load_dataset_info(dataset):
    try:
        info = DATASET_INFO[dataset]
    except KeyError:
        info = {
            'name': dataset,
            'url': '' # TODO later update with paper URL
        }
    return info