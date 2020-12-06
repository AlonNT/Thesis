import argparse
import datetime
import json
import os
import shutil
import sys
import wandb

import torch
from loguru import logger

from model import MainNetDGL, get_cnn
from utils import get_data, train_model


def main():
    args = parse_args()

    if args.device.startswith('cuda:') and not torch.cuda.is_available():
        raise ValueError("CUDA is not available, yet \'device\' was given as \'cuda:i\'")

    logger_format = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>'

    datetime_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = os.path.join(args.path, datetime_string)
    os.mkdir(out_dir)
    # shutil.copyfile(args.params_json, os.path.join(out_dir, os.path.basename(args.params_json)))

    # Configure the logger:
    # (1) Remove the default logger (to stdout) and use a one with a custom format.
    # (2) Adds a log file named `run.log` in the given output directory.
    logger.remove()
    logger.remove()
    logger.add(sink=sys.stdout, format=logger_format)
    logger.add(sink=os.path.join(out_dir, 'run.log'), format=logger_format)

    device = torch.device(args.device)
    # assert (params['DNI'] is None) or (params['DGL'] is None), 'Both DNI and DGL is not supported'

    # model = MainNetDNI(params['architecture'], params['DNI'], params['DGL']).to(device)
    model: torch.nn.Module = MainNetDGL() if args.dgl else get_cnn()
    model: torch.nn.Module = model.to(device)

    # Define a Loss function and optimizer.
    # We use a Classification Cross-Entropy loss,
    # and SGD with momentum and weight_decay.
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_params = dict(lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
    image_datasets, dataloaders, dataset_sizes = get_data(args.batch_size)

    logger.info(f'Starting to train {args.epochs} epochs (on {args.device}) | '
                f'batch-size={args.batch_size}, '
                f'learning-rate={args.learning_rate}, '
                f'weight-decay={args.weight_decay}, '
                f'momentum={args.momentum}')

    # wandb.watch(model, criterion=criterion)
    wandb.init(project='bgd', config=args)
    wandb.watch(model)

    train_model(model, criterion, optimizer_params, dataloaders, dataset_sizes, device, args.epochs,
                log_interval=100, is_dgl=args.dgl)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Beyond Gradient-Descent main script.'
    )

    parser.add_argument('-p', '--path', type=str, default='experiments',
                        help='The path to output the log file summarizing the experiment.')
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='On which device to execute the training (\'cpu\' or \'cuda:i\').')
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-05)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0)
    parser.add_argument('-m', '--momentum', type=float, default=0.9)
    parser.add_argument('--dgl', action='store_true',
                        help='Use decoupled greedy learning.')

    return parser.parse_args()


if __name__ == '__main__':
    main()
