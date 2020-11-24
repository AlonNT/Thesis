import argparse
import datetime
import json
import os
import shutil
import sys

import torch
from loguru import logger

from model import MainNet
from utils import get_data, train_model


def main():
    args = parse_args()

    with open(args.params_json, 'r') as f:
        params = json.load(f)

    if params['device'].startswith('cuda:') and not torch.cuda.is_available():
        raise ValueError("CUDA is not available, yet \'device\' was given as \'cuda:i\'")

    logger_format = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>'

    out_dir = os.path.join(params['path'], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.mkdir(out_dir)
    shutil.copyfile(args.params_json, os.path.join(out_dir, os.path.basename(args.params_json)))

    # Configure the logger:
    # (1) Remove the default logger (to stdout) and use a one with a custom format.
    # (2) Adds a log file named `run.log` in the given output directory.
    logger.remove()
    logger.remove()
    logger.add(sink=sys.stdout, format=logger_format)
    logger.add(sink=os.path.join(out_dir, 'run.log'), format=logger_format)

    device = torch.device(params['device'])
    assert (params['DNI'] is None) or (params['DGL'] is None), 'Both DNI and DGL is not supported'

    model = MainNet(params['architecture'], params['DNI'], params['DGL']).to(device)

    # Define a Loss function and optimizer.
    # We use a Classification Cross-Entropy loss,
    # and SGD with momentum and weight_decay.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), **params['optimizer_params'])
    image_datasets, dataloaders, dataset_sizes = get_data()

    train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, params['epochs'])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Beyond Gradient-Descent main script'
    )

    parser.add_argument('-p', '--params_json', default='./params.json', type=str,
                        help='This is the params JSON that should contain all the relevant arguments.')

    return parser.parse_args()


if __name__ == '__main__':
    main()
