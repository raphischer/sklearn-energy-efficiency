import os
import base64

import numpy as np
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import black
import fitz # PyMuPDF

from mlel.ratings import calculate_compound_rating, load_results, rate_results


C_SIZE = (1560, 2411)
POS_TEXT = {
    # infos that are directly taken from summary via keys
    "name":                                     ('drawString',        90, '-Bold', .04,  .855, None),
    "task_type":                                ('drawString',        90, '',      .04,  .815, None),
    "environment":                              ('drawString',        68, '',      .04,  .42,  None),
    "dataset":                                  ('drawRightString',   90, '',      .95,  .815, None),
    "inference_power_draw":                     ('drawRightString',   68, '-Bold', .25,  .28,  None),
    "inference_time":                           ('drawRightString',   68, '-Bold', .75,  .25,  None),
    "parameters":                               ('drawRightString',   68, '-Bold', .744,  .05,  '{} M /'),
    "gflops":                                   ('drawString',        68, '-Bold', .77, .05,  '{} B'),
    "train_time":                               ('drawRightString',   68, '-Bold', .7,  .25,  '{} /'),
    "train_time_epoch":                         ('drawString',        68, '-Bold', .715, .25,  None),
    "train_power_draw":                         ('drawRightString',   68, '-Bold', .2,  .28,  '{} /'),
    "train_power_draw_epoch":                   ('drawString',        68, '-Bold', .215,  .28,  None),
    "top1_val":                                 ('drawRightString',   68, '-Bold', .22,  .05,  '{} /'),
    "top5_val":                                 ('drawString',        68, '-Bold', .235, .05,  None),
    # infos that are extracted via methods
    'format_power_draw_sources':                ('drawCentredString', 56, '',      .25,  .22,  None),
    # static infos, depending on $task
    "$Inference Power Draw per Sample":         ('drawCentredString', 56, '',      .25,  .25,  None),
    "$Inference Runtime per Sample":            ('drawCentredString', 56, '',      .75,  .22,  None),
    "$Inference Top-1 / Top-5 Accuracy":        ('drawCentredString', 56, '',      .25,  .02,  None),
    "$Inference Parameters / Flops":            ('drawCentredString', 56, '',      .75,  .02,  None),
    "$Inference Ws":                            ('drawString',        56, '',      .27,  .28,  None),
    "$Inference ms":                            ('drawString',        56, '',      .77,  .25,  None),
    "$Inference [%]":                           ('drawString',        56, '',      .34,  .05,  None),
    "$Training Power Draw Total / per Epoch":   ('drawCentredString', 56, '',      .25,  .25,  None),
    "$Training Runtime Total / per Epoch":      ('drawCentredString', 56, '',      .75,  .22,  None),
    "$Training Top-1 / Top-5 Accuracy":         ('drawCentredString', 56, '',      .25,  .02,  None),
    "$Training kWh":                            ('drawString',        56, '',      .31,  .28,  None),
    "$Training h":                              ('drawString',        56, '',      .81,  .25,  None),
    "$Training [%]":                            ('drawString',        56, '',      .34,  .05,  None),
    "$Training Parameters / Flops":             ('drawCentredString', 56, '',      .75,  .02,  None),
}
POS_RATINGS = { char: (.66, y) for char, y in zip('ABCDE', reversed(np.linspace(.461, .727, 5))) }
ICON_NAME_TO_METRIC = {
    'Inference': {
        'time': 'inference_time',
        'top': 'top1_val',
        'power_draw': 'inference_power_draw',
        'parameters': 'parameters'
    },
    'Training': {
        'time': 'train_time',
        'top': 'top1_val',
        'power_draw': 'train_power_draw',
        'parameters': 'parameters'
    }
}
ICON_POS = {
    'time': (1050, 680),
    'top': (250, 200),
    'power_draw': (250, 770),
    'parameters': (1050, 200)
}
PARTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "label_design", "parts")


def format_power_draw_sources(summary):
    sources = 'Sources:'
    for key, vals in summary['power_draw_sources'].items():
        if len(vals) > 0:
            sources += f' {key},'
    return sources[:-1]


class EnergyLabel(fitz.Document):

    def __init__(self, summary, rating_mode):
        canvas = Canvas("result.pdf", pagesize=C_SIZE)
        frate = calculate_compound_rating(summary, rating_mode, 'ABCDE')
        # Background
        canvas.drawInlineImage(os.path.join(PARTS_DIR, f"bg.png"), 0, 0)
        # Rated Pictograms
        for icon, (posx, posy) in ICON_POS.items():
            metric = ICON_NAME_TO_METRIC[summary['task_type']][icon]
            rating = summary[metric]['rating']
            canvas.drawInlineImage(os.path.join(PARTS_DIR, f"{icon}_{rating}.png"), posx, posy)
        # Final Rating
        canvas.drawInlineImage(os.path.join(PARTS_DIR, f"Rating_{frate}.png"), POS_RATINGS[frate][0] * C_SIZE[0], POS_RATINGS[frate][1] * C_SIZE[1])
        # Add stroke to make even bigger letters
        canvas.setFillColor(black)
        canvas.setLineWidth(3)
        canvas.setStrokeColor(black)
        text=canvas.beginText()
        text.setTextRenderMode(2)
        canvas._code.append(text.getCode())
        # Text parts
        for key, (draw_method, fsize, style, x, y, fmt) in POS_TEXT.items():
            draw_method = getattr(canvas, draw_method)
            canvas.setFont('Helvetica' + style, fsize)
            if key in globals():
                text = globals()[key](summary)
            elif key.startswith(f"${summary['task_type']} "):
                # Static text on label depending on the task type
                text = key.replace(f"${summary['task_type']} ", "")
            elif key in summary:
                # Dynamic text that receives content from summary
                if isinstance(summary[key], dict):
                    text = 'n.a.' if summary[key]["value"] is None else f'{summary[key]["value"]:4.2f}'[:4]
                    if text.endswith('.'):
                        text = text[:-1]
                else:
                    text = summary[key]
            else:
                text = None
            if text is not None:
                if fmt is not None:
                    text = fmt.format(text)
                draw_method(int(C_SIZE[0] * x), int(C_SIZE[1] * y), text)
        super().__init__(stream=canvas.getpdfdata(), filetype='pdf')
    
    def to_encoded_image(self):
        label_bytes = self.load_page(0).get_pixmap().tobytes()
        base64_enc = base64.b64encode(label_bytes).decode('ascii')
        return 'data:image/png;base64,{}'.format(base64_enc)


if __name__ == "__main__":
    _, summaries = load_results('results')
    summaries, _, _ = rate_results(summaries, 'ResNet101')
    test_summaries = [
        summaries['RTX 5000 - TensorFlow 2.4.1']['inference'][0],
        summaries['RTX 5000 - TensorFlow 2.4.1']['inference'][3],
        summaries['RTX 5000 - TensorFlow 2.4.1']['inference'][6],
        # summaries['A100 x8 - Torch 1.10.2+cu113']['training'][0]
    ]
    for idx, summary in enumerate(test_summaries):
        pdf_doc = EnergyLabel(summary, 'mean')
        pdf_doc.save(f'testlabel_{idx}.pdf')