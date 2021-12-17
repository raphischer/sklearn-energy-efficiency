import argparse
from collections import namedtuple
import datetime
import json
import os
import time
import sys

from load_imagenet import load_imagenet, resize_with_crop
from util import fix_seed, create_output_dir, Logger, prepare_model, set_gpu, prepare_optimizer, start_monitoring


def main(args):
    args.gpu = set_gpu(args.gpu)

    args.seed = fix_seed(args.seed)

    with open(os.path.join(args.model_dir, 'config.json'), 'r') as m_cfg:
        cfg = json.load(m_cfg)
        cfg.update(args.__dict__)

    args.output_dir = create_output_dir(os.path.join(args.output_dir, 'eval'), args.use_timestamp_dir, cfg)

    args = namedtuple('CFG', cfg)(**cfg)

    # reroute the stdout to logfile, remember to call close!
    sys.stdout = Logger(os.path.join(args.output_dir, 'logfile.txt'))

    dataset, ds_info = load_imagenet(args.data_path, None, args.split, resize_with_crop, args.batch_size, args.n_batches)
    optimizer = prepare_optimizer(args.model, args.opt.lower(), args.lr, args.momentum, args.weight_decay, ds_info, args.epochs)
    model = prepare_model(args.model, optimizer, weights=args.model_dir)

    print("Start evaluation")
    start_time = time.time()

    monitoring = start_monitoring(args.gpu_monitor_interval, args.cpu_monitor_interval, args.output_dir, args.gpu)

    eval_result = model.evaluate(dataset)

    for monitor in monitoring:
        monitor.stop()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Evaluation time {total_time_str}")

    print(dict(zip(model.metrics_names, eval_result)))

    return eval_result

    
def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description="Classification training with Tensorflow, based on PyTorch training", add_help=add_help)

    parser.add_argument("--data-path", default="/raid/imagenet", type=str, help="dataset path")
    parser.add_argument("--split", default="train", choices=['train', 'validation'], type=str, help="dataset split to use")
    parser.add_argument("--gpu-monitor-interval", default=1, type=float, help="Setting to > 0 activates GPU profiling every X seconds")
    parser.add_argument("--cpu-monitor-interval", default=1, type=float, help="Setting to > 0 activates CPU profiling every X seconds")
    parser.add_argument("--n-batches", default=-1, type=int, help="number of batches to take")
    parser.add_argument("-b", "--batch-size", default=256, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--model-dir", default="/raid/fischer/dnns/train_2021_12_15_15_01", type=str, help="path to access the trained model")
    parser.add_argument("--output-dir", default="/raid/fischer/dnns", type=str, help="path to save outputs")
    parser.add_argument("--seed", type=int, default=-1, help="Seed to use (if -1, uses and logs random seed)"),
    parser.add_argument("--use-timestamp-dir", default=True, action="store_true", help="Creates timestamp directory in data path")
    parser.add_argument("--gpu", default=0, type=int, help="gpu to use for computations (if available)")
    
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    eval_result = main(args)