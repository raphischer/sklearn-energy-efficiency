import os
import json
import base64

import numpy as np


KEYS = ["parameters", "fsize", "power_draw", "inference_time", "top1_val", "top5_val"]
HIGHER_BETTER = [
    'top1_val',
    'top5_val',
]
BACKENDS = {
    'tensorflow': ('TensorFlow', 'tensorflow'),
    'pytorch': ('Torch', 'torch'),
}
GPU_NAMES = {
    'NVIDIA A100-SXM4-40GB': 'A100',
    'Quadro RTX 5000': 'RTX 5000',
}


def get_environment_key(log):
    backend_name, pip_name = BACKENDS[log['config']['backend']]
    backend_version = [r.split('==')[1] for r in log['requirements'] if r.split('==')[0] == pip_name][0]
    n_gpus = len(log['execution_platform']['GPU'])
    gpu_name = GPU_NAMES[log['execution_platform']['GPU']['0']['Name']]
    gpu_str = f'{gpu_name} x{n_gpus}' if n_gpus > 1 else gpu_name
    return f'{gpu_str} - {backend_name} {backend_version}'


def aggregate_rating(ratings, mode, meanings=None):
    if isinstance(ratings, dict): # model summary given instead of list of ratings
        ratings = [val['rating'] for val in ratings.values() if 'rating' in val]
    if meanings is None:
        meanings = np.arange(np.max(ratings) + 1)
    if mode == 'best':
        return meanings[min(ratings)]
    if mode == 'worst':
        return meanings[max(ratings)]
    if mode == 'median':
        return meanings[int(np.median(ratings))]
    if mode == 'mean':
        return meanings[int(np.ceil(np.mean(ratings)))]
    if mode == 'majority':
        return meanings[np.argmax(np.bincount(ratings))]
    raise NotImplementedError('Rating Mode not implemented!', mode)


def value_to_index(value, ref, metric_key):
    # TODO If values is integer, just return integer
    #      i = v / r                     OR                i = r / v
    return value / ref if metric_key in HIGHER_BETTER else ref / value


def index_to_value(index, ref, metric_key):
    if index == 0:
        index = 10e-4
    #      v = i * r                            OR          v = r / i
    return index * ref  if metric_key in HIGHER_BETTER else ref / index


def calculate_rating(index, scale):
    for i, (upper, lower) in enumerate(scale):
        if index <= upper and index > lower:
            return i


def calc_accuracy(res, train=False, top5=False):
    split = 'train' if train else 'validation'
    metric = 'top_5_accuracy' if top5 else 'accuracy'
    return res[split]['results']['metrics'][metric]


def calc_parameters(res):
    return res['validation']['results']['model']['params'] * 1e-6


def calc_fsize(res):
    return res['validation']['results']['model']['fsize']


def calc_inf_time(res):
    return res['train']['duration'] / 1281167 * 1000


def calc_power_draw(res):
    return res['train']["monitoring_gpu"]["total"]["total_power_draw"] / 1281167


def load_scale(content="mlel/scales.json"):
    if isinstance(content, dict):
        scales_json = content
    elif isinstance(content, str):
        with open(content, "r") as file:
            scales_json = json.load(file)

    # Convert boundaries to dictionary
    max_value = 100
    min_value = 0

    scale_intervals = {}

    for key in KEYS:
        boundaries = scales_json[key]
        intervals = [[max_value, boundaries[0]]]
        for i in range(len(boundaries)-1):
            intervals.append([boundaries[i], boundaries[i+1]])
        intervals.append([boundaries[-1], min_value])
        
        scale_intervals[key] = intervals

    return scale_intervals


def save_scale(scale_intervals, output="scales.json"):
    scale = {}
    for key in KEYS:
        scale[key] = [sc[0] for sc in scale_intervals[key][1:]]

    if output is not None:
        with open(output, 'w') as out:
            json.dump(scale, out, indent=4)
    
    return json.dumps(scale, indent=4)


def load_results(results_directory):
    logs = {}

    for fname in os.listdir(results_directory):
        with open(os.path.join(results_directory, fname), 'r') as rf:
            log = json.load(rf)
            env_key = get_environment_key(log)
            if env_key not in logs:
                logs[env_key] = {}
            if log['config']['model'] in logs[env_key]:
                raise NotImplementedError(f'Already found results for {log["config"]["model"]} on {env_key}, averaging runs is not implemented (yet)!')
            logs[env_key][log['config']['model']] = log

    # Exctract all relevant metadata
    summaries = {}
    for env_key, env_logs in logs.items():
        for model_name, model_log in env_logs.items():
            model_information = {'environment': env_key, 'name': model_name, 'dataset': 'ImageNet'}
            model_information['parameters'] = {'value': calc_parameters(model_log)}
            model_information['fsize'] = {'value': calc_fsize(model_log)}
            model_information['power_draw'] = {'value': calc_power_draw(model_log)}
            model_information['inference_time'] = {'value': calc_inf_time(model_log)}
            model_information['top1_val'] = {'value': calc_accuracy(model_log)}
            model_information['top5_val'] = {'value': calc_accuracy(model_log, top5=True)}

            try:
                summaries[env_key].append(model_information)
            except Exception:
                summaries[env_key] = [model_information]

    # Transform logs dict for one environment to list of logs
    for env_key, env_logs in logs.items():
        logs[env_key] = [model_logs for model_logs in env_logs.values()]

    return logs, summaries


def rate_results(summaries, reference_name, scales=None):
    if scales is None:
        scales = load_scale()    

    # Get reference values
    reference_values = {}
    for env_key, env_logs in summaries.items():
        for model in env_logs:
            if model['name'] == reference_name:
                reference_values[env_key] = {k: v['value'] for k, v in model.items() if isinstance(v, dict) }
                break

    # Calculate indices using reference values and scales
    for env_key, env_logs in summaries.items():
        for model in env_logs:
            for key in KEYS:
                model[key]['index'] = value_to_index(model[key]['value'], reference_values[env_key][key], key)
                model[key]['rating'] = calculate_rating(model[key]['index'], scales[key])

    # Calculate the real-valued scales
    real_scales = {}
    for env, ref_values in reference_values.items():
        real_scales[env] = {}
        for key, vals in scales.items():
            real_scales[env][key] = [(index_to_value(start, ref_values[key], key), index_to_value(stop, ref_values[key], key)) for (start, stop) in vals]
    
    return summaries, scales, real_scales
