# importing package
import os
import time
import json

import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc

from mlee.ratings import calculate_compound_rating, METRICS_INFO
from mlee.label_generator import EnergyLabel
from elex import Visualization
from mlee.unit_reformatting import CustomUnitReformater
from mlee.meta_info import load_dataset_info
from evaluate_hyperparams import classifiers

# SETUP

app = Visualization('results', external_stylesheets=[dbc.themes.BOOTSTRAP])
os.chdir('paper_results')
PLOT_WIDTH = 250
RATINGS = ['A', 'B', 'C', 'D', 'E']
RA = '#639B30'
RB = '#B8AC2B'
RC = '#F8B830'
RD = '#EF7D29'
RE = '#E52421'
COLORS = [RA, RB, RC, RD, RE]
ENVNAME = 'Xeon(R) W-2155 - Scikit-learn 1.1.2'
CUR = CustomUnitReformater()
SEL_DS = 'covtype'


# DATA SHAPES

ds_sizes = []
for ds in app.datasets:
    shapes = app.logs[ds]['inference'][ENVNAME][0]['validation']['results']['data']['shape']
    shape = (shapes['train'][0] + shapes['test'][0], shapes['train'][1])
    ds_sizes.append((shape[0], ds))
    print(f'{ds:<35} - Shape {shape}')
BIGGEST_DS = [ds for _, ds in reversed(sorted(ds_sizes))]


# #####################################
# #### FIND SOLID REFERENCE MODELS ####
# #####################################
# for ds, ds_logs in app.summaries.items():
#     for task, task_logs in ds_logs.items():
#         keys = ["general top1_val", f"{task} power_draw", "general parameters", f"{task} time"]
#         for env, env_logs in task_logs.items():
#             models = []
#             for model in env_logs:
#                 perf = ' - '.join([f'{model[key]["fmt_val"]} {model[key]["fmt_unit"]}' for key in keys])
#                 models.append({key: model[key]["value"] for key in keys})
#                 models[-1]['name'] = model["name"]
#                 # model_string = f'{ds[:20]:<20} {task:<10} {env:<30} {model["name"]:<20} {perf}'
#                 # models[-1]['text'] = model_string
#             model_performance = {key: sorted([(model[key], model['name']) for model in models]) for key in keys}
#             num_avgs = [1, 2, 3, 4, 5, 6]
#             for num_avg in num_avgs:
#                 idx_start = int(len(models) / 2 - num_avg / 2)
#                 idx_end = idx_start + num_avg
#                 avg_models = {}
#                 for key, models in model_performance.items():
#                     avg_models[key] = set([mod for _, mod in models[idx_start:idx_end]])
#                 union_set = set.intersection(*list(avg_models.values()))
#                 if len(union_set) > 0:
#                     print(f'{ds[:20]:<20} {task:<10} {env:<30} {str(union_set)}')
#                     break


# BEST MODEL TABLE

TEX_TABLE_INDEXING_RESULTS = r'''
    \begin{tabular}{l|l|c|c|c|c|c|c|c|c}
        \toprule 
        \multirow{2}{*}{Dataset} & \multirow{2}{*}{Method}  & \multicolumn{2}{c}{$METR1} & \multicolumn{2}{c}{$METR2} & \multicolumn{2}{c}{$METR3} & \multicolumn{2}{c}{$METR4}  \\  \cline{3-10}
        & & $VALIND \\ 
        \midrule
        $RES
        \bottomrule
    \end{tabular}'''
TABLE_NAMES = {
    "inference power_draw": ("Power Draw", 'Ws'),
    "general f1_val": ("$F_1$ Score", '%'),
    "inference time": ("Running Time", 'ms'),
    "general parameters": ("Parameters", '#'),
}
            
final_text = TEX_TABLE_INDEXING_RESULTS
rows = []
for ds in BIGGEST_DS:
    results = app.summaries[ds]['inference'][ENVNAME]
    res_ratings = [calculate_compound_rating(res, app.rating_mode) for res in results]
    res_acc = [res['general f1_val']['value'] if res_ratings[r_i] == min(res_ratings) else 0 for r_i, res in enumerate(results)]
    best_mod_ind = np.argsort(res_acc)[-1]
    best_mod, rating = results[best_mod_ind], res_ratings[best_mod_ind]

    mod_name_cc = r'\colorbox{R' + RATINGS[rating] + r'}{' + RATINGS[rating] + '} ' + classifiers[best_mod['name']][0]
    model_res = [load_dataset_info(ds)['name'], mod_name_cc]
    val_ind_txt = []
    for m_i, (metr, (name, unit_to)) in enumerate(TABLE_NAMES.items()):
        final_text = final_text.replace(f'$METR{m_i + 1}', name)
        val_ind_txt.append(f'[{unit_to}] & Index')
        ind = r'\colorbox{R' + RATINGS[best_mod[metr]['rating']] + r'}{' + f"{best_mod[metr]['index']:5.3f}"[:5] + '}'
        val = CUR.reformat_value(best_mod[metr]['value'], METRICS_INFO[metr][2], unit_to)[0]
        if '+' not in val:
            val = val[:5]
        elif '+0' in val:
            val = val.replace('+0', '+')
        model_res.extend([val, ind])
    final_text = final_text.replace('$VALIND', ' & '.join(val_ind_txt))
    rows.append(' & '.join(model_res) + r' \\')
final_text = final_text.replace('$RES', '\n        '.join(rows))
final_text = final_text.replace('%', '\%')
final_text = final_text.replace('#', '\#')
with open('table_best_models.tex', 'w') as outf:
    outf.write(final_text)


# METRIC WEIGHTING TABLE

table_str = r'''
\begin{tabular}{l|c}
    \toprule 
    Metric & Weight \\
    \midrule
    $ROWS
    \bottomrule
\end{tabular}
\begin{tikzpicture}[remember picture, overlay]
    $ARROWS
\end{tikzpicture}'''
categories = {}
# extract all metrics, weights and categories
with open(os.path.join('..', 'mlee', 'weighting.json'), 'r') as wfile:
    weights = json.load(wfile)
    for metric, weight in weights.items():
        name = METRICS_INFO[metric][0]
        category = METRICS_INFO[metric][1]
        if weight > 0 and 'training' not in metric:
            if category not in categories:
                categories[category] = []
            categories[category].append((weight, name))
rows, arrows, ri = [], [], 0
for cat, metrics in categories.items():
    rs = ri
    for (m_weight, m_name) in reversed(sorted(metrics)):
        rows.append(r'\tikzmark{' + str(ri) + r'}' + f'{m_name} & {m_weight}' + r'\\')
        ri += 1
    arrows.append(r'\draw[brace mirrored, thick] ($(pic cs:' + str(rs) + r') + (-3pt, 6pt)$)--($(pic cs:' + str(ri - 1) + r') + (-3pt, 0)$) node [midway, left] {' + cat + r'};')

table_str = table_str.replace('$ROWS', '\n    '.join(rows))
table_str = table_str.replace('$ARROWS', '\n    '.join(arrows))
with open('table_weights.tex', 'w') as outf:
    outf.write(table_str)


# DUMMY OUTPUT for setting up pdf export of plotly

fig=px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.write_image("dummy.pdf")
time.sleep(2)
os.remove("dummy.pdf")


# SCATTER PLOTS

app.dataset = SEL_DS
app.task = 'inference'
app.xaxis = 'inference power_draw'
app.yaxis = 'general f1_val'

fig = app.update_scatter_graph()
fig.update_layout(xaxis_range=[0, 1.7])#, yaxis_range=[0.94, 1.035])
fig.update_layout(width=PLOT_WIDTH * 2, height=300)
fig.update_traces(textposition=f'middle right', textfont_size=18)
fig.write_image('scatter index.pdf')

# fig = app.update_scatter_graph(scale_switch='value')
# fig.update_layout(width=PLOT_WIDTH, height=350)
# fig.write_image('scatter value.pdf')


# EXAMPLARY LABELS

for mod in app.summaries[SEL_DS]['inference'][ENVNAME]:
    label = EnergyLabel(mod, app.rating_mode)
    label.save(f'label {mod["name"]}.pdf')


# BAR PLOTS DATASETS

for d_i, ds in enumerate(BIGGEST_DS):
    app.dataset = ds
    app.update_scatter_graph()
    fig = app.update_bars_graph(discard_y_axis=True)
    fig.update_layout(xaxis_title=load_dataset_info(app.dataset)['name'])
    fig.update_layout(width=PLOT_WIDTH, height=100)
    fig.write_image(f'distrib {str(d_i).zfill(2)}.pdf')

# BAR PLOTS MODELS

for m_i, mod in enumerate(app.summaries['olivetti_faces']['inference'][ENVNAME]):
    model = mod['name']
    plot_data = {'': {'ratings': []}}
    for ds in app.datasets:
        results = app.summaries[ds]['inference'][ENVNAME]
        for res in results:
            if res['name'] == model:
                plot_data['']['ratings'].append(calculate_compound_rating(res, app.rating_mode))
                break
    app.plot_data = plot_data
    fig = app.update_bars_graph(discard_y_axis=True)
    fig.update_layout(xaxis_title=model)
    fig.update_layout(width=PLOT_WIDTH, height=100)
    fig.write_image(f'model distrib {str(m_i).zfill(2)}.pdf')
