import argparse
import datetime
import os
import sys
import wandb

import torch
from loguru import logger

from basic_cnn import CNNwDGL, CNNwDNI
from vgg import VGG, VGGwDGL
from utils import get_dataloaders, train_model, get_cnn


def get_model(args):
    if args.model == "CNN":
        if args.dgl:
            return CNNwDGL(), 'CNN with DGL'
        elif args.dni:
            return CNNwDNI(), 'CNN with DNI'
        elif args.cdni:
            return CNNwDNI(use_context=True), 'CNN with cDNI'
        else:
            return get_cnn(), 'CNN'
    elif args.model.startswith('VGG'):
        kwargs = dict(vgg_name=args.model,
                      aux_mlp_n_hidden_layers=args.aux_mlp_n_hidden_layers,
                      aux_mlp_hidden_dim=args.aux_mlp_hidden_dim,
                      dropout_prob=args.dropout_prob)
        if args.dgl:
            model_name = f'{args.model} with DGL'
            if args.ssl:
                kwargs['use_ssl'] = True
                model_name += ' & SSL'
                if args.upsample:
                    kwargs['upsample'] = True
                    model_name += ' (upsampling)'

            return VGGwDGL(**kwargs), model_name
        else:
            return VGG(**kwargs), args.model
    else:
        raise ValueError(f'Illegal \'model\': {args.model}.')


def main():
    args = parse_args()

    if args.device.startswith('cuda:') and not torch.cuda.is_available():
        raise ValueError("CUDA is not available, yet \'device\' was given as \'cuda:i\'.")

    logger_format = '<magenta>{time:YYYY-MM-DD HH:mm:ss}</magenta> | <level>{message}</level>'

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

    model, model_name = get_model(args)

    logger.info(f'Starting to train {model_name} '
                f'for {args.epochs} epochs '
                f'(using device {args.device}) | '
                f'optimizer-type={args.optimizer_type}, '
                f'batch-size={args.batch_size}, '
                f'learning-rate={args.learning_rate}, '
                f'weight-decay={args.weight_decay}, '
                f'momentum={args.momentum}',
                f'log-interval={args.log_interval}')

    device = torch.device(args.device)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    ssl_criterion = torch.nn.L1Loss().to(device)
    dataloaders = get_dataloaders(args.batch_size,
                                  normalize_to_unit_gaussian=args.enable_normalization_to_unit_gaussian,
                                  normalize_to_plus_minus_one=not args.disable_normalization_to_plus_minus_one,
                                  random_crop=not args.disable_random_crop,
                                  random_horizontal_flip=not args.disable_random_horizontal_flip,
                                  random_erasing=args.enable_random_erasing)

    wandb.init(project='thesis', config=args)
    wandb.watch(model)

    optimizer_params = dict(optimizer_type=args.optimizer_type, lr=args.learning_rate,
                            weight_decay=args.weight_decay, momentum=args.momentum)
    train_model(model, criterion, optimizer_params, dataloaders, device,
                args.epochs, args.log_interval, args.dgl, args.cdni,
                args.ssl, ssl_criterion, args.pred_loss_weight, args.ssl_loss_weight,
                args.first_trainable_block)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Main script. '
                    'This enables running the different experiments while logging to a log-file and to wandb.'
    )

    default_model = 'VGG16'
    default_path = 'experiments'
    default_device = 'cpu'
    default_epochs = 1500
    default_optimizer = 'SGD'
    default_batch_size = 64
    default_learning_rate = 0.001
    default_weight_decay = 0
    default_momentum = 0.9
    default_dropout_prob = 0
    default_log_interval = 100
    default_aux_mlp_n_hidden_layers = 1
    default_aux_mlp_hidden_dim = 1024
    default_pred_loss_weight = 1
    default_ssl_loss_weight = 0.1
    default_first_trainable_block = 0

    parser.add_argument('--model', type=str, default=default_model,
                        help=f'The model name for the network architecture. '
                             f'Default is {default_model}.')
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
                        help=f'Momentum (will be used only if optimizer-type is SGD). '
                             f'Default is {default_momentum}.')
    parser.add_argument('--dropout_prob', type=float, default=default_dropout_prob,
                        help=f'Dropout probability (will be added after each non linearity). '
                             f'Default is {default_dropout_prob}.')
    parser.add_argument('--log_interval', type=int, default=default_log_interval,
                        help=f'How many iterations between each training log. '
                             f'Default is {default_log_interval}.')
    parser.add_argument('--aux_mlp_n_hidden_layers', type=int, default=default_aux_mlp_n_hidden_layers,
                        help=f'How many hidden layers in each auxiliary network (which is a MLP). '
                             f'Default is {default_aux_mlp_n_hidden_layers}.')
    parser.add_argument('--aux_mlp_hidden_dim', type=int, default=default_aux_mlp_hidden_dim,
                        help=f'Dimension of each hidden layer in each auxiliary network (which is a MLP). '
                             f'Default is {default_aux_mlp_hidden_dim}.')
    parser.add_argument('--dgl', action='store_true',
                        help='Use decoupled greedy learning.')
    parser.add_argument('--dni', action='store_true',
                        help='Use decoupled neural interfaces.')
    parser.add_argument('--cdni', action='store_true',
                        help='Use decoupled neural interfaces with context.')
    parser.add_argument('--ssl', action='store_true',
                        help='Use self-supervised local loss (predict shifted image).')
    parser.add_argument('--upsample', action='store_true',
                        help='Whether to upsample SSL predictions or leave them downsampled.')
    parser.add_argument('--pred_loss_weight', type=float, default=default_pred_loss_weight,
                        help=f'Weight of the prediction loss when using both prediction-loss and SSL-loss.'
                             f'Default is {default_pred_loss_weight}.')
    parser.add_argument('--ssl_loss_weight', type=float, default=default_ssl_loss_weight,
                        help=f'Weight of the SSL loss when using both prediction-loss and SSL-loss.'
                             f'Default is {default_ssl_loss_weight}.')
    parser.add_argument('--first_trainable_block', type=int, default=default_first_trainable_block,
                        help=f'The first trainable block index, can be used to fix first few blocks initial weights.'
                             f'Default is {default_first_trainable_block}.')

    parser.add_argument('--disable_normalization_to_plus_minus_one', action='store_true',
                        help='If true, disable normalization of the values to the range [-1,1] (instead of [0,1]).')
    parser.add_argument('--disable_random_crop', action='store_true',
                        help='If true, disable random cropping which is padding of 4 followed by random crop.')
    parser.add_argument('--disable_random_horizontal_flip', action='store_true',
                        help='If true, disable random horizontal flip.')
    parser.add_argument('--enable_normalization_to_unit_gaussian', action='store_true',
                        help='If true, enable normalization of the values to a unit gaussian.')
    parser.add_argument('--enable_random_erasing', action='store_true',
                        help='If true, performs erase a random rectangle in the image.')

    return parser.parse_args()


if __name__ == '__main__':
    main()
