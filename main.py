import argparse
import datetime
import os
import sys
import wandb

import torch
from loguru import logger

from model import MainNetDGL, MainNetDNI, get_cnn
from utils import get_data, train_model


def main():
    args = parse_args()

    if args.device.startswith('cuda:') and not torch.cuda.is_available():
        raise ValueError("CUDA is not available, yet \'device\' was given as \'cuda:i\'.")

    logger_format = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>'

    datetime_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = os.path.join(args.path, datetime_string)
    os.mkdir(out_dir)

    # Configure the logger:
    # (1) Remove the default logger (to stdout) and use a one with a custom format.
    # (2) Adds a log file named `run.log` in the given output directory.
    logger.remove()
    logger.remove()
    logger.add(sink=sys.stdout, format=logger_format)
    logger.add(sink=os.path.join(out_dir, 'run.log'), format=logger_format)

    if args.dni:
        model_str = 'CNN with DNI'
    elif args.cdni:
        model_str = 'CNN with cDNI'
    elif args.dgl:
        model_str = 'CNN with DGL'
    else:
        model_str = 'Regular CNN'
    logger.info(f'Starting to train {model_str} '
                f'for {args.epochs} epochs '
                f'(using device {args.device}) | '
                f'optimizer-type={args.optimizer_type}, '
                f'batch-size={args.batch_size}, '
                f'learning-rate={args.learning_rate}, '
                f'weight-decay={args.weight_decay}, '
                f'momentum={args.momentum}',
                f'log-interval={args.log_interval}')

    if args.dgl:
        model = MainNetDGL()
    elif args.dni:
        model = MainNetDNI()
    elif args.cdni:
        model = MainNetDNI(use_context=True)
    else:
        model = get_cnn()

    device = torch.device(args.device)
    model = model.to(device)

    # Define a Loss function and optimizer.
    # We use a Classification Cross-Entropy loss,
    # and SGD with momentum and weight_decay.
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_params = dict(optimizer_type=args.optimizer_type, lr=args.learning_rate,
                            weight_decay=args.weight_decay, momentum=args.momentum)
    image_datasets, dataloaders, dataset_sizes = get_data(args.batch_size)

    wandb.init(project='bgd', config=args)
    wandb.watch(model)

    train_model(model, criterion, optimizer_params, dataloaders, dataset_sizes, device, args.epochs,
                log_interval=args.log_interval, is_dgl=args.dgl, is_cdni=args.cdni)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Beyond back-propagation playground main script. '
                    'This enables running the different experiments while logging to a log-file and to wandb.'
    )

    default_path = 'experiments'
    default_device = 'cpu'
    default_epochs = 1500
    default_optimizer = 'Adam'
    default_batch_size = 256
    default_learning_rate = 3e-05
    default_weight_decay = 0
    default_momentum = 0.9
    default_log_interval = 100

    parser.add_argument('--path', type=str, default=default_path,
                        help=f'Output path for the experiment - '
                             f'a sub-directory named with the data and time will be created within. '
                             f'Default is {default_path}.')
    parser.add_argument('--device', type=str, default=default_device,
                        help=f'\'cpu\' or \'cuda:i\'. '
                             f'Default is {default_device}.')
    parser.add_argument('--epochs', type=int, default=default_epochs,
                        help=f'Number of epochs. '
                             f'Default is {default_epochs}).')
    parser.add_argument('--optimizer_type', type=str, default=default_optimizer,
                        help=f'Optimizer type - SGD or Adam. '
                             f'Default is {default_optimizer}.')
    parser.add_argument('--batch_size', type=int, default=default_batch_size,
                        help=f'Batch size. '
                             f'Default is {default_batch_size}.')
    parser.add_argument('--learning_rate', type=float, default=default_learning_rate,
                        help=f'Learning-rate. '
                             f'Default is {default_learning_rate}.')
    parser.add_argument('--weight_decay', type=float, default=default_weight_decay,
                        help=f'Weight decay. '
                             f'Default is {default_weight_decay}.')
    parser.add_argument('--momentum', type=float, default=default_momentum,
                        help=f'Momentum (will be used only if optimizer-type is SGD).'
                             f'Default is {default_momentum}.')
    parser.add_argument('--log_interval', type=int, default=default_log_interval,
                        help=f'How many iterations between each training log. '
                             f'Default is {default_log_interval}.')
    parser.add_argument('--dgl', action='store_true',
                        help='Use decoupled greedy learning.')
    parser.add_argument('--dni', action='store_true',
                        help='Use decoupled neural interfaces.')
    parser.add_argument('--cdni', action='store_true',
                        help='Use decoupled neural interfaces with context.')

    return parser.parse_args()


if __name__ == '__main__':
    main()
