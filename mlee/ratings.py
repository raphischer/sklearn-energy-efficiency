import os
import json

import numpy as np

from mlee.meta_info import HARDWARE_NAMES, TASK_TYPES, load_model_info, load_dataset_info
from mlee.unit_reformatting import CustomUnitReformater


HIGHER_BETTER = [
    'general top1_val',
    'general top5_val',
]
METRICS_INFO = {
    'general parameters':     ('Parameters',                    'number',      lambda model_log, m_info: calc_parameters(model_log, m_info)),
    'general flops':          ('Floating Point Operations',     'number',      lambda model_log, m_info: calc_flops(model_log, m_info)),
    'general fsize':          ('Model File Size',               'bytes',       lambda model_log, m_info: calc_fsize(model_log, m_info)),
    'general top1_val':       ('Top-1 Validation Accuracy',     'percent',     lambda model_log, m_info: calc_accuracy(model_log, m_info)),
    'general top5_val':       ('Top-5 Validation Accuracy',     'percent',     lambda model_log, m_info: calc_accuracy(model_log, m_info, top5=True)),
    'inference power_draw':   ('Power Draw / Sample',  'wattseconds', lambda model_log, m_info: calc_power_draw(model_log, m_info)),
    'inference time':         ('Time / Sample',        'seconds',     lambda model_log, m_info: calc_inf_time(model_log, m_info)),
    # 'training power_draw_epoch': ('Training Power Draw per Epoch', 'wattseconds', lambda model_log, m_info: calc_power_draw_train(model_log, m_info, True)),
    'training power_draw':       ('Total Power Draw',      'wattseconds', lambda model_log, m_info: calc_power_draw_train(model_log, m_info)),
    # 'training time_epoch':       ('Training Time per Epoch',       'seconds',     lambda model_log, m_info: calc_time_train(model_log, m_info, True)),
    'training time':             ('Total Time',            'seconds',     lambda model_log, m_info: calc_time_train(model_log, m_info))        
}
DEFAULT_REFERENCES = {
    'imagenet': 'ResNet101',
    'olivetti_faces': 'SGD',
    'lfw_people': 'Extra Random Forest',
    '20newsgroups_vectorized': 'Random Forest',
    'covtype': 'AdaBoost',
    'lfw_pairs': 'Random Forest',
    'kddcup99': 'Nearest',
    'credit-g': 'Extra Random Forest',
    'mnist_784': 'Random Forest',
    'SpeedDating': 'Extra Random Forest',
    'phoneme': 'SGD',
    'blood-transfusion-service-center': 'SGD'
}



def load_backend_info(backend): # TODO access from train / infer scripts, and log info during experiment
    info_fname = os.path.join(os.path.dirname(__file__), f'ml_{backend}', 'info.json')
    with open(info_fname, 'r') as info_f:
        return json.load(info_f)


def get_environment_key(log):
    backend = load_backend_info(log['config']['backend'])
    backend_version = 'n.a.'
    for package in backend["Packages"]:
        for req in log['requirements']:
            if req.split('==')[0].replace('-', '_') == package.replace('-', '_'):
                backend_version = req.split('==')[1]
                break
        else:
            continue
        break
    n_gpus = len(log['execution_platform']['GPU'])
    if len(log['execution_platform']['GPU']) > 0:
        gpu_name = HARDWARE_NAMES[log['execution_platform']['GPU']['0']['Name']]
        name = f'{gpu_name} x{n_gpus}' if n_gpus > 1 else gpu_name
    else:
        name = HARDWARE_NAMES[log['execution_platform']['Processor']]
    return f'{name} - {backend["Name"]} {backend_version}'


def calculate_compound_rating(ratings, mode, meanings=None):
    if isinstance(ratings, dict): # model summary given instead of list of ratings
        weights = [val['weight'] for val in ratings.values() if isinstance(val, dict) and 'rating' in val if val['weight'] > 0]
        weights = [w / sum(weights) for w in weights]
        ratings = [val['rating'] for val in ratings.values() if isinstance(val, dict) and 'rating' in val if val['weight'] > 0]
    else:
        weights = [1.0 / len(ratings) for _ in ratings]
    if meanings is None:
        meanings = np.arange(np.max(ratings) + 1, dtype=int)
    round_m = np.ceil if 'pessimistic' in mode else np.floor # optimistic
    if mode == 'best':
        return meanings[min(ratings)] # TODO no weighting here
    if mode == 'worst':
        return meanings[max(ratings)] # TODO no weighting here
    if 'median' in mode:
        asort = np.argsort(ratings)
        weights = np.array(weights)[asort]
        ratings = np.array(ratings)[asort]
        cumw = np.cumsum(weights)
        for i, (cw, r) in enumerate(zip(cumw, ratings)):
            if cw == 0.5:
                return meanings[int(round_m(np.average([r, ratings[i + 1]])))]
            if cw > 0.5 or (cw < 0.5 and cumw[i + 1] > 0.5):
                return meanings[r]
    if 'mean' in mode:
        return meanings[int(round_m(np.average(ratings, weights=weights)))]
    if mode == 'majority':
        return meanings[np.argmax(np.bincount(ratings))]
    raise NotImplementedError('Rating Mode not implemented!', mode)


def value_to_index(value, ref, metric_key):
    #      i = v / r                     OR                i = r / v
    try:
        return value / ref if metric_key in HIGHER_BETTER else ref / value
    except:
        return 0


def index_to_value(index, ref, metric_key):
    if index == 0:
        index = 10e-4
    #      v = i * r                            OR         v = r / i
    return index * ref  if metric_key in HIGHER_BETTER else ref / index


def index_to_rating(index, scale):
    for i, (upper, lower) in enumerate(scale):
        if index <= upper and index > lower:
            return i
    return 4 # worst rating if index does not fall in boundaries


def calc_accuracy(res, model_info, train=False, top5=False):
    split = 'train' if train else 'validation'
    metric = 'top_5_accuracy' if top5 else 'accuracy'
    return res[split]['results']['metrics'][metric] * 100


def calc_parameters(res, model_info):
    if 'validation' in res:
        return res['validation']['results']['model']['params']
    return res['results']['model']['params']


def calc_flops(res, model_info):
    if 'validation' in res:
        return res['validation']['results']['model']['flops']
    return res['results']['model']['flops']


def calc_fsize(res, model_info):
    if 'validation' in res:
        return res['validation']['results']['model']['fsize']
    return res['results']['model']['fsize']


def calc_inf_time(res, model_info):
    return res['validation']['duration'] # TODO / 50000


def calc_power_draw(res, model_info):
    power_draw = 0
    if res['validation']["monitoring_pynvml"] is not None:
        power_draw += res['validation']["monitoring_pynvml"]["total"]["total_power_draw"]
    if res['validation']["monitoring_pyrapl"] is not None:
        power_draw += res['validation']["monitoring_pyrapl"]["total"]["total_power_draw"]
    return power_draw # TODO/ 50000


def calc_time_train(res, model_info, per_epoch=False):
    if len(res['results']['history']) < 1:
        if per_epoch:
            return None
        else:
            return res["duration"]
    val_per_epoch = res["duration"] / len(res["results"]["history"]["loss"])
    if not per_epoch:
        val_per_epoch *= model_info['model_info']['epochs']
    return val_per_epoch


def calc_power_draw_train(res, model_info, per_epoch=False):
    power_draw = 0
    if res["monitoring_pynvml"] is not None:
        power_draw += res["monitoring_pynvml"]["total"]["total_power_draw"]
    if res["monitoring_pyrapl"] is not None:
        power_draw += res["monitoring_pyrapl"]["total"]["total_power_draw"]
    # if there is no information on training epochs
    if len(res['results']['history']) < 1:
        if per_epoch:
            return None
        else:
            return power_draw
    val_per_epoch = power_draw / len(res["results"]["history"]["loss"])
    if not per_epoch:
        val_per_epoch *= model_info['model_info']['epochs']
    return val_per_epoch


def characterize_monitoring(summary):
    sources = {
        'GPU': ['NVML'] if summary['monitoring_pynvml'] is not None else [],
        'CPU': ['RAPL'] if summary['monitoring_pyrapl'] is not None else [],
        'Extern': []
    }
    # TODO also make use of summary['monitoring_psutil']
    # if summary['monitoring_psutil'] is not None:
    #     sources['CPU'].append('psutil')
    return sources


def calculate_optimal_boundaries(summaries, quantiles):
    boundaries = {}
    for task, sum_task in summaries.items():
        for metric in METRICS_INFO[task].keys():
            index_values = [ env_sum[metric]['index'] for env_sums in sum_task.values() for env_sum in env_sums if env_sum[metric]['index'] is not None ]
            try:
                boundaries[metric] = np.quantile(index_values, quantiles)
            except Exception as e:
                print(e)
    return load_boundaries(boundaries)


def load_boundaries(content="mlee/boundaries.json"):
    if isinstance(content, dict):
        boundary_json = content
    elif isinstance(content, str):
        with open(content, "r") as file:
            boundary_json = json.load(file)

    # Convert boundaries to dictionary
    max_value = 10000
    min_value = 0

    boundary_intervals = {}

    for key, boundaries in boundary_json.items():
        intervals = [[max_value, boundaries[0]]]
        for i in range(len(boundaries)-1):
            intervals.append([boundaries[i], boundaries[i+1]])
        intervals.append([boundaries[-1], min_value])
        
        boundary_intervals[key] = intervals

    return boundary_intervals


def save_boundaries(boundary_intervals, output="boundaries.json"):
    scale = {}
    for key in boundary_intervals.keys():
        scale[key] = [sc[0] for sc in boundary_intervals[key][1:]]

    if output is not None:
        with open(output, 'w') as out:
            json.dump(scale, out, indent=4)
    
    return json.dumps(scale, indent=4)


def save_weights(summaries, output="weights.json"):
    weights = {}
    for task_summaries in summaries.values():
        any_summary = list(task_summaries.values())[0][0]
        for key, vals in any_summary.items():
            if isinstance(vals, dict) and 'weight' in vals:
                weights[key] = vals['weight']
    if output is not None:
        with open(output, 'w') as out:
            json.dump(weights, out, indent=4)
    
    return json.dumps(weights, indent=4)


def update_weights(summaries, weights, axis=None):
    for task_summaries in summaries.values():
        for env_summaries in task_summaries.values():
            for model_sum in env_summaries:
                if isinstance(weights, dict):
                    for key, values in model_sum.items():
                        if key in weights:
                            values['weight'] = weights[key]
                else: # only update a single metric weight
                    if axis in model_sum:
                        model_sum[axis]['weight'] = weights
    return summaries


def load_results(results_directory, weighting=None):
    if weighting is None:
        with open(os.path.join(os.path.dirname(__file__), 'weighting.json'), 'r') as wf:
            weighting = json.load(wf)
    
    logs, summaries = {}, {}
    for subdir in os.listdir(results_directory):
        subdir = os.path.join(results_directory, subdir)
        if os.path.isdir(subdir):
            sub_logs, sub_summaries = load_subresults(subdir, weighting)
            logs.update(sub_logs)
            summaries.update(sub_summaries)
        else:
            raise RuntimeError('Found non-directory item in results folder!')
    return logs, summaries


def load_subresults(results_subdir, weighting):    
    logs = {}
    for fname in os.listdir(results_subdir):
        with open(os.path.join(results_subdir, fname), 'r') as rf:
            log = json.load(rf)
            env_key = get_environment_key(log)
            task_type = TASK_TYPES[fname.split('_')[0]]
            dataset = log['config']['dataset'] if 'dataset' in log['config'] else os.path.basename(results_subdir)
            if dataset not in logs:
                logs[dataset] = {task: {} for task in TASK_TYPES.values()}
            if env_key not in logs[dataset][task_type]:
                logs[dataset][task_type][env_key] = {}
            if log['config']['model'] in logs[dataset][task_type][env_key]:
                raise NotImplementedError(f'Already found results for {log["config"]["model"]} on {env_key}, averaging runs is not implemented (yet)!')
            if 'dataset' not in log['config']: # TODO remove later (when ImageNet results have been rerun)
                log['config']['dataset'] = 'imagenet'
            logs[dataset][task_type][env_key][log['config']['model']] = log

    # Exctract all relevant metadata
    summaries = {ds: {task: {} for task in TASK_TYPES.values()} for ds in logs.keys()}
    
    unit_fmt = CustomUnitReformater()
    for dataset, ds_logs in logs.items():
        for task_type, task_logs in ds_logs.items():
            for env_key, env_logs in task_logs.items():
                if env_key not in summaries[dataset][task_type]:
                    summaries[dataset][task_type][env_key] = []
            
                # general model information
                for model_name, model_log in env_logs.items():
                    model_information = {
                        'environment': env_key,
                        'name': model_name,
                        'model_info': load_model_info(model_log['config']['dataset'], model_name),
                        'dataset_info': load_dataset_info(model_log['config']['dataset']),
                        'task_type': task_type.capitalize(),
                        'power_draw_sources': characterize_monitoring(model_log if 'monitoring_pynvml' in model_log else model_log['validation'])
                    }
                    # Calculate metrics for rating
                    for metrics_key, (metric_name, unit, metrics_calculation) in METRICS_INFO.items():
                        if metrics_key.split()[0] in ['general', task_type]:
                            model_information[metrics_key] = {'name': metric_name, 'unit': unit, 'weight': weighting[metrics_key]}
                            try:
                                try:
                                    value = metrics_calculation(model_log, model_information)
                                except KeyError: # the top-1 accuracy needs to be extracted from the inference logs
                                    model_inference_log = logs[dataset]['inference'][env_key][model_name]
                                    value = metrics_calculation(model_inference_log, model_information)
                                fmt_val, fmt_unit = unit_fmt.reformat_value(value, unit)
                                model_information[metrics_key].update({'value': value, 'fmt_val': fmt_val, 'fmt_unit': fmt_unit})
                            except Exception:
                                model_information[metrics_key].update({'value': None, 'fmt_val': 'n.a.', 'fmt_unit': ''})
                    summaries[dataset][task_type][env_key].append(model_information)

    # Transform logs dict for one environment to list of logs
    for ds, ds_logs in logs.items():
        for task_type, task_logs in ds_logs.items():
            for env_key, env_logs in task_logs.items():
                logs[ds][task_type][env_key] = [model_logs for model_logs in env_logs.values()]

    return logs, summaries


def rate_results(summaries, references=None, boundaries=None):
    if references is None:
        references = DEFAULT_REFERENCES
    if boundaries is None:
        boundaries = load_boundaries()

    # Get reference values for each dataset, environment and task
    reference_values = {}
    for ds, ds_logs in summaries.items():
        reference_name = references[ds]
        reference_values[ds] = {}
        for task_type, task_logs in ds_logs.items():
            reference_values[ds][task_type] = {env_key: {} for env_key in task_logs.keys()}
            for env_key, env_logs in task_logs.items():
                for model in env_logs:
                    if model['name'] == reference_name:
                        for metrics_key, metrics_val in model.items():
                            if isinstance(metrics_val, dict) and 'value' in metrics_val:
                                if metrics_val['value'] is None:
                                    raise RuntimeError(f'Found unratable metric {metrics_key} for reference model {reference_name} on {env_key} {task_type}!')
                                reference_values[ds][task_type][env_key][metrics_key] = metrics_val['value']
                        break
                else:
                    raise RuntimeError(f'Reference {reference_name} not found in {ds} {task_type} {env_key} logs!')
            
            # Calculate value indices using reference values and boundaries
            for env_key, env_summs in summaries[ds][task_type].items():
                for model in env_summs:
                    for key in model.keys():
                        if isinstance(model[key], dict) and 'value' in model[key]:
                            if model[key]['value'] is None:
                                model[key]['index'] = None
                                model[key]['rating'] = 4
                            else:
                                model[key]['index'] = value_to_index(model[key]['value'], reference_values[ds][task_type][env_key][key], key)
                                model[key]['rating'] = index_to_rating(model[key]['index'], boundaries[key])

    # Calculate the real-valued boundaries
    real_boundaries = {}
    for ds, ds_ref_values in reference_values.items():
        real_boundaries[ds] = {}
        for task_type, task_ref_values in ds_ref_values.items():
            real_boundaries[ds][task_type] = {env_key: {} for env_key in task_ref_values.keys()}
            for env_key, env_ref_values in task_ref_values.items():
                for key, val in env_ref_values.items():
                    real_boundaries[ds][task_type][env_key][key] = [(index_to_value(start, val, key), index_to_value(stop, val, key)) for (start, stop) in boundaries[key]]
    
    return summaries, boundaries, real_boundaries
