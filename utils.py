import argparse
import math
import copy
import os
import sys
import itertools
import warnings
import wandb
import yaml
import torch
import torchvision

import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

from pathlib import Path
from functools import partial
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from typing import Any, List, Literal, Optional, Callable, Tuple, Union, Type
from loguru import logger
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from torchvision.transforms import (ToTensor, RandomCrop, RandomResizedCrop, RandomHorizontalFlip, Normalize, Compose,
                                    Resize)
from torchvision.transforms.functional import resize
from torchvision.datasets import ImageFolder
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.trainer.states import RunningStage

from consts import LOGGER_FORMAT
from schemas.architecture import ArchitectureArgs
from schemas.data import DataArgs
from schemas.environment import EnvironmentArgs
from schemas.optimization import OptimizationArgs
from vgg import get_vgg_blocks, configs, get_vgg_model_kernel_size


def log_args(args):
    """Logs the given arguments to the logger's output.
    """
    logger.info(f'Running with the following arguments:')
    longest_arg_name_length = max(len(k) for k in args.flattened_dict().keys())
    pad_length = longest_arg_name_length + 4
    for arg_name, value in args.flattened_dict().items():
        logger.info(f'{f"{arg_name} ":-<{pad_length}} {value}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Main script for running the experiments with arguments from the corresponding pydantic schema'
    )
    parser.add_argument('--yaml_path', help=f'(Optional) path to a YAML file with the arguments')
    return parser.parse_known_args()


def get_args(args_class):
    """Gets arguments as an instance of the given pydantic class,
    according to the argparse object (possibly including the yaml config).
    """
    known_args, unknown_args = parse_args()
    args_dict = None
    if known_args.yaml_path is not None:
        with open(known_args.yaml_path, 'r') as f:
            args_dict = yaml.load(f, Loader=yaml.FullLoader)

    if args_dict is None:  # This happens when the yaml file is empty, or no yaml file was given.
        args_dict = dict()

    while len(unknown_args) > 0:
        arg_name = unknown_args.pop(0).replace('--', '')
        values = list()
        while (len(unknown_args) > 0) and (not unknown_args[0].startswith('--')):
            values.append(unknown_args.pop(0))
        if len(values) == 0:
            raise ValueError(f'Argument {arg_name} given in command line has no corresponding value.')
        value = values[0] if len(values) == 1 else values

        categories = list(args_class.__fields__.keys())
        found = False
        for category in categories:
            category_args = list(args_class.__fields__[category].default.__fields__.keys())
            if arg_name in category_args:
                if category not in args_dict:
                    args_dict[category] = dict()
                args_dict[category][arg_name] = value
                found = True

        if not found:
            raise ValueError(f'Argument {arg_name} is not recognized.')

    args = args_class.parse_obj(args_dict)

    return args


def get_possibly_sparse_linear_layer(in_features: int, out_features: int, sparse_fraction: float):
    if sparse_fraction > 0:
        return RandomlySparseConnected(in_features, out_features, sparse_fraction)
    else:
        return nn.Linear(in_features, out_features)


def get_mlp(input_dim: int,
            output_dim: int,
            n_hidden_layers: int = 0,
            hidden_dimensions: Union[int, List[int]] = 0,
            use_batch_norm: bool = False,
            organize_as_blocks: bool = True,
            shuffle_blocks_output: Union[bool, List[bool]] = False,
            fixed_permutation_per_block: bool = False,
            sparse_fractions: Optional[List[float]] = None) -> torch.nn.Sequential:
    """Create an MLP (i.e. Multi-Layer-Perceptron) and return it as a PyTorch's sequential model.

    Args:
        input_dim: The dimension of the input tensor.
        output_dim: The dimension of the output tensor.
        n_hidden_layers: Number of hidden layers.
        hidden_dimensions: The dimension of each hidden layer.
        use_batch_norm: Whether to use BatchNormalization after each layer or not.
        organize_as_blocks: Whether to organize the model as blocks of Linear->(BatchNorm)->ReLU.
        shuffle_blocks_output: If it's true - shuffle the output of each block in the network.
            If it's a list of values, define as single value which will be True if any one of the values is True.
        fixed_permutation_per_block: If it's true - use a fixed permutation per block in the network
            and not sample a new one each time.
        sparse_fractions: If given (i.e. it's not None) should be a list of floats,
            indicating the sparsity of each layer. A number 0 < q < 1 indicates that
            only a q fraction of the neurons will be connected. 
            Zero means that a fully-connected will be used.
    Returns:
        A sequential model which is the constructed MLP.
    """
    layers: List[torch.nn.Module] = list()
    if isinstance(shuffle_blocks_output, list):
        shuffle_blocks_output = any(shuffle_blocks_output)
    if not isinstance(hidden_dimensions, list):
        hidden_dimensions = [hidden_dimensions] * n_hidden_layers
    assert len(hidden_dimensions) == n_hidden_layers
    sparse_fractions = get_list_of_arguments(sparse_fractions, len(hidden_dimensions) + 1, default=0)

    in_features = input_dim
    for i, hidden_dim in enumerate(hidden_dimensions):
        block_layers: List[nn.Module] = list()
        out_features = hidden_dim

        # Begins with a `Flatten` layer. It's useful when the input is 4D from a conv layer, and harmless otherwise.
        if i == 0:
            block_layers.append(nn.Flatten())

        block_layers.append(get_possibly_sparse_linear_layer(in_features, out_features, sparse_fractions[i]))
        if use_batch_norm:
            block_layers.append(torch.nn.BatchNorm1d(hidden_dim))
        block_layers.append(torch.nn.ReLU())

        if shuffle_blocks_output:
            block_layers.append(ShuffleTensor(spatial_size=1, channels=out_features,
                                              fixed_permutation=fixed_permutation_per_block))

        if organize_as_blocks:
            layers.append(nn.Sequential(*block_layers))
        else:
            layers.extend(block_layers)

        in_features = out_features

    final_layer = get_possibly_sparse_linear_layer(in_features, output_dim, sparse_fractions[-1])
    if organize_as_blocks:
        block_layers = [final_layer]
        if len(hidden_dimensions) == 0:
            block_layers = [nn.Flatten()] + block_layers
        layers.append(nn.Sequential(*block_layers))
    else:
        if len(hidden_dimensions) == 0:
            layers.append(nn.Flatten())
        layers.append(final_layer)

    return nn.Sequential(*layers)


def get_list_of_arguments(arg, length, default=None):
    if isinstance(arg, list):
        assert len(arg) == length
        return copy.deepcopy(arg)
    else:
        if (default is not None) and (arg is None):
            if isinstance(default, list):
                return copy.deepcopy(default)
            arg = default
        return [arg] * length


class View(nn.Module):
    def __init__(self, shape: tuple):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor):
        return x.view(x.shape[0], *self.shape)

    def extra_repr(self) -> str:
        return f'shape={self.shape}'


# # For debugging purposes, e.g. to verify the gradients of the weights 
# # in the locations where mask is 0 remain unchanged.
# def verify_grad_is_zero_where_mask_is_0(grad: torch.Tensor, mask: torch.Tensor):
#     # For debugging purposes, e.g. to verify the gradients of the weights 
#     # in the locations where mask is 0 remain unchanged.
#     assert torch.all((grad == 0) | (mask == 1)).item()


class RandomlySparseConnected(nn.Module):
    def __init__(self, in_features: int, out_features: int, fraction: float,
                 bias: bool = True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.fraction = fraction
        self.num_nonzero_weights_per_output_neuron = int(round(fraction * in_features))

        self.mask = nn.Parameter(torch.zeros((out_features, in_features), **factory_kwargs), requires_grad=False)
        self.weight = nn.Parameter(torch.zeros((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        # # For debugging purposes, e.g. to verify the gradients of the weights 
        # # in the locations where mask is 0 remain unchanged.
        # self.weight.register_hook(functools.partial(verify_grad_is_zero_where_mask_is_0,  mask=self.mask))

    def reset_parameters(self) -> None:
        """Initializes the parameters if the module, which are
        the random mask, and the weight and bias of the linear layer.
        """
        self.init_random_mask()
        self.init_weight_and_bias()

    def init_random_mask(self):
        """Initializes the boolean mask indicating the (sparse) connections between the input and output neurons.

        The mask is a tensor of shape (self.out_features, self.in_features) and dtype np.float32,
        where a value of 1 in the coordinate ij means that the i-th output neuron 
        is connected to the j-th input neuron.
        """
        ordered_indices_vector = np.tile(np.arange(self.in_features), self.out_features)
        ordered_indices_matrix = ordered_indices_vector.reshape(self.out_features, self.in_features)
        shuffled_indices_matrix = np.random.default_rng().permuted(ordered_indices_matrix, axis=1)
        indices = shuffled_indices_matrix[:, :self.num_nonzero_weights_per_output_neuron]
        for i in range(self.out_features):
            self.mask[i, indices[i]] = 1

    @torch.no_grad()
    def init_weight_and_bias(self):
        """Initializes the weight matrix and bias vector of the linear layer.
        
        The weight matrix and bias vector of the linear layer are initialized as if they were
        a linear layer from `self.num_nonzero_weights_per_output_neuron` input neurons to 
        `self.out_features` output neurons (since de facto this is what's going to happen).

        To understand why this is done in the context-manager `no_grad` see here:
        https://medium.com/@mrityu.jha/understanding-the-grad-of-autograd-fc8d266fd6cf
        Prevents the RuntimeError:
            "RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation."
        """
        tmp_linear = nn.Linear(in_features=self.num_nonzero_weights_per_output_neuron,
                               out_features=self.out_features,
                               bias=(self.bias is not None),
                               device=self.weight.device)
        bool_mask = self.mask.bool()
        for i in range(self.out_features):
            self.weight[i, bool_mask[i]] = tmp_linear.weight.data[i]
        self.bias = tmp_linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

    def extra_repr(self) -> str:
        return f'in_feature={self.in_features}, ' + \
               f'out_features={self.out_features}, ' + \
               f'bias={self.bias is not None}, ' + \
               f'fraction={self.fraction:.3f}'


def get_cnn(conv_channels: List[int],
            linear_channels: List[int],
            kernel_sizes: Optional[List[int]] = None,
            strides: Optional[List[int]] = None,
            paddings: Optional[List[int]] = None,
            use_max_pool: Optional[List[bool]] = None,
            shuffle_outputs: Optional[List[bool]] = None,
            spatial_only: Optional[List[bool]] = None,
            fixed_permutation: Optional[List[bool]] = None,
            replace_with_linear: Optional[List[bool]] = None,
            replace_with_bottleneck: Optional[List[int]] = None,
            randomly_sparse_connected_fractions: Optional[List[float]] = None,
            adaptive_avg_pool_before_mlp: bool = False,
            max_pool_after_first_conv: bool = False,
            in_spatial_size: int = 32,
            in_channels: int = 3,
            n_classes: int = 10) -> torch.nn.Sequential:
    """
    This function builds a CNN and return it as a PyTorch's sequential model.

    :param conv_channels: A list of integers containing the channels of each convolution block.
                                 Each block will contain Conv - BatchNorm - MaxPool - ReLU.
    :param linear_channels: A list of integers containing the channels of each linear hidden layer.
    :param kernel_sizes: The kernel size to use in each conv layer.
    :param strides: The stride to use in each conv layer.
    :param paddings: The amount of padding to use in each conv layer.
    :param use_max_pool: Whether to use max-pooling in each layer of the network or not.
    :param shuffle_outputs: Whether to shuffle the output of each layer or not.
    :param spatial_only: The argument to pass to `ShuffleTensor` (see doc there).
    :param fixed_permutation: The argument to pass to `ShuffleTensor` (see doc there).
    :param replace_with_linear: Whether to replace each conv layer with a linear layer of the same expressiveness.
    :param replace_with_bottleneck: Whether to replace each conv layer with a "bottleneck" linear layer 
        of the same expressiveness, meaning a linear layer of low rank constraint (e.g. 100,000 -> 1,000 -> 100,000).
        The number represent the middle linear layer dimensionality.
    :param in_spatial_size: Will be used to infer input dimension for the first affine layer.
    :param in_channels: Number of channels in the input tensor.
    :param n_classes: Number of classes (i.e. determines the size of the prediction vector 
        containing the classes' scores).
    :return: A sequential model which is the constructed CNN.
    """
    blocks: List[nn.Sequential] = list()

    use_max_pool = get_list_of_arguments(
        use_max_pool, len(conv_channels), default=False)
    shuffle_outputs = get_list_of_arguments(
        shuffle_outputs, len(conv_channels), default=False)
    strides = get_list_of_arguments(
        strides, len(conv_channels), default=1)
    kernel_sizes = get_list_of_arguments(
        kernel_sizes, len(conv_channels), default=3)
    paddings = get_list_of_arguments(
        paddings, len(conv_channels), default=[kernel_size // 2 for kernel_size in kernel_sizes])
    spatial_only_list = get_list_of_arguments(
        spatial_only, len(conv_channels), default=True)
    fixed_permutation_list = get_list_of_arguments(
        fixed_permutation, len(conv_channels), default=True)
    replace_with_linear = get_list_of_arguments(
        replace_with_linear, len(conv_channels), default=False)
    replace_with_bottleneck = get_list_of_arguments(
        replace_with_bottleneck, len(conv_channels), default=0)
    randomly_sparse_connected_fractions = get_list_of_arguments(
        randomly_sparse_connected_fractions, len(conv_channels) + len(linear_channels) + 1, default=0)

    zipped_args = zip(conv_channels, paddings, strides, kernel_sizes, use_max_pool,
                      shuffle_outputs, spatial_only_list, fixed_permutation_list,
                      replace_with_linear, replace_with_bottleneck,
                      randomly_sparse_connected_fractions[:len(conv_channels)])
    for i, (out_channels, padding, stride, kernel_size, pool,
            shuf, spatial, fixed,
            linear, bottleneck, sparse_fraction) in enumerate(zipped_args):
        block_layers: List[nn.Module] = list()

        out_spatial_size = int(math.floor((in_spatial_size + 2 * padding - kernel_size) / stride + 1))
        if pool:
            out_spatial_size = int(math.floor(out_spatial_size / 2))

        in_features = in_channels * (in_spatial_size ** 2)
        out_features = out_channels * (out_spatial_size ** 2)

        if bottleneck > 0:
            assert not (linear or (sparse_fraction > 0)), \
                'When bottleneck is greater than 0, both linear and sparse_fraction should be turned off.'
            block_layers.append(nn.Flatten())
            block_layers.append(nn.Linear(in_features, bottleneck))
            block_layers.append(nn.Linear(bottleneck, out_features))
            block_layers.append(View(shape=(out_channels, out_spatial_size, out_spatial_size)))
        elif linear or (sparse_fraction > 0):
            assert not (linear and (sparse_fraction > 0)), \
                'Select one of linear/sparse-linear, not both.'
            block_layers.append(nn.Flatten())
            block_layers.append(get_possibly_sparse_linear_layer(in_features, out_features, sparse_fraction))
            block_layers.append(View(shape=(out_channels, out_spatial_size, out_spatial_size)))
        else:
            block_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))

        block_layers.append(nn.BatchNorm2d(out_channels))
        block_layers.append(nn.ReLU())

        if pool:
            block_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if shuf:
            block_layers.append(ShuffleTensor(out_spatial_size, out_channels, spatial, fixed))
        if max_pool_after_first_conv and (i == 0):
            block_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        if adaptive_avg_pool_before_mlp and (i == len(conv_channels) - 1):
            block_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            out_spatial_size = 1

        blocks.append(nn.Sequential(*block_layers))
        in_channels = out_channels
        in_spatial_size = out_spatial_size

    mlp = get_mlp(input_dim=in_channels * (in_spatial_size ** 2),
                  output_dim=n_classes,
                  n_hidden_layers=len(linear_channels),
                  hidden_dimensions=linear_channels,
                  use_batch_norm=True,
                  organize_as_blocks=True,
                  sparse_fractions=randomly_sparse_connected_fractions[len(conv_channels):])
    blocks.extend(list(mlp))

    return torch.nn.Sequential(*blocks)


@torch.no_grad()
def calc_aggregated_patch(dataloader,
                          patch_size,
                          agg_func: Callable,
                          existing_model: Optional[nn.Module] = None):
    """Calculate the aggregated patch across all patches in the dataloader.

    Args:
        dataloader: dataloader to iterate on.
        patch_size: The patch-size to feed into the aggregate function.
        agg_func: The aggregate function, which gets a single argument which is a NumPy array,
            and return a single argument which is a NumPy array
        existing_model: An (optionally) existing model to call on each image in the data.
    """
    total_size = 0
    mean = None
    device = get_model_device(existing_model)
    for inputs, _ in tqdm(dataloader, total=len(dataloader), desc='Calculating mean patch'):
        inputs = inputs.to(device)
        if existing_model is not None:
            inputs = existing_model(inputs)

        # Unfold the input batch to its patches - shape (N, C*H*W, M) where M is the number of patches per image.
        patches = F.unfold(inputs, patch_size)

        # Transpose to (N, M, C*H*W) and then reshape to (N*M, C*H*W) to have collection of vectors
        # Also make contiguous in memory
        patches = patches.transpose(1, 2).flatten(0, 1).contiguous().double()

        # Perform the aggregation function over the batch-size and number of patches per image.
        # For example, when calculating mean it'll a (C*H*W)-dimensional vector,
        # and when calculating the covariance it will be a square matrix of shape (C*H*W, C*H*W)
        aggregated_patch = agg_func(patches)

        if mean is None:
            mean = torch.zeros_like(aggregated_patch)

        batch_size = inputs.size(0)
        mean = ((total_size / (total_size + batch_size)) * mean +
                (batch_size / (total_size + batch_size)) * aggregated_patch)

        total_size += batch_size

    return mean


def calc_covariance(data, mean=None):
    """Calculates the covariance-matrix of the given data.

    This function assumes the data matrix is ordered as rows-vectors
    (i.e. shape (n,d) so n data-points in d dimensions).

    Args:
        data: The given data, a 2-dimensional NumPy array ordered as rows-vectors
            (i.e. shape (n,d) so n data-points in d dimensions).
        mean: The mean of the data, if not given the mean will be calculated.
            It's useful when the mean is the mean of some larger distribution, and not only the mean of the
            given data array (as done when calculating the covariance matrix of the whole patches distribution).

    Returns:
        The covariance-matrix of the given data.
    """
    if mean is None:
        mean = data.mean(axis=0)
    centered_data = data - mean
    return (1 / data.shape[0]) * (centered_data.T @ centered_data)


def calc_whitening_from_dataloader(dataloader: DataLoader,
                                   patch_size: int,
                                   whitening_regularization_factor: float,
                                   zca_whitening: bool = False,
                                   existing_model: Optional[nn.Module] = None) -> np.ndarray:
    """Calculates the whitening matrix from the given data.

    Denote the data matrix by X (i.e. collection of patches) with shape N x D.
    N is the number of patches, and D is the dimension of each patch (channels * spatial_size ** 2).
    This function returns the whitening operator as a columns-vectors matrix of shape D x D,
    so it needs to be multiplied by the target data matrix X' of shape N' x D from the right (X' @ W)
    [and NOT from the left, i.e. NOT W @ X'].

    Args:
        dataloader: The given data to iterate on.
        patch_size: The size of the patches to calculate the whitening on.
        whitening_regularization_factor: The regularization factor used when calculating the whitening,
            which is some small constant positive float added to the denominator.
        zca_whitening: Whether it's ZCA whitening (or PCA whitening).
        existing_model: An (optionally) existing model to call on each image in the data.

    Returns:
        The whitening matrix.
    """
    logger.debug('Performing a first pass over the dataset to calculate the mean patch...')
    mean_patch = calc_aggregated_patch(dataloader, patch_size, agg_func=partial(torch.mean, dim=0),
                                       existing_model=existing_model)

    logger.debug('Performing a second pass over the dataset to calculate the covariance...')
    covariance_matrix = calc_aggregated_patch(dataloader, patch_size,
                                              agg_func=partial(calc_covariance, mean=mean_patch),
                                              existing_model=existing_model)

    logger.debug('Calculating eigenvalues decomposition to get the whitening matrix...')
    whitening_matrix = get_whitening_matrix_from_covariance_matrix(
        covariance_matrix.cpu(), whitening_regularization_factor, zca_whitening
    )

    logger.debug('Done.')
    return whitening_matrix


def configure_logger(out_dir: str, level='INFO', print_sink=sys.stdout):
    """
    Configure the logger:
    (1) Remove the default logger (to stdout) and use a one with a custom format.
    (2) Adds a log file named `run.log` in the given output directory.
    """
    logger.remove()
    logger.remove()
    logger.add(sink=print_sink, format=LOGGER_FORMAT, level=level)
    logger.add(sink=os.path.join(out_dir, 'run.log'), format=LOGGER_FORMAT, level=level)


def get_dataloaders(batch_size: int = 64,
                    normalize_to_unit_gaussian: bool = False,
                    normalize_to_plus_minus_one: bool = False,
                    random_crop: bool = False,
                    random_horizontal_flip: bool = False,
                    random_erasing: bool = False,
                    random_resized_crop: bool = False):
    """Gets dataloaders for the CIFAR10 dataset, including data augmentations as requested by the arguments.

    Args:
        batch_size: The size of the mini-batches to initialize the dataloaders.
        normalize_to_unit_gaussian: If true, normalize the values to be a unit gaussian.
        normalize_to_plus_minus_one: If true, normalize the values to be in the range [-1,1] (instead of [0,1]).
        random_crop: If true, performs padding of 4 followed by random crop.
        random_horizontal_flip: If true, performs random horizontal flip.
        random_erasing: If true, erase a random rectangle in the image. See https://arxiv.org/pdf/1708.04896.pdf.
        random_resized_crop: If true, performs random resized crop.

    Returns:
        A dictionary mapping "train"/"test" to its dataloader.
    """
    raise NotImplementedError('This function is deprecated and will be removed in the future.')

    transforms = {'train': list(), 'test': list()}

    if random_horizontal_flip:
        transforms['train'].append(torchvision.transforms.RandomHorizontalFlip())
    if random_crop:
        transforms['train'].append(torchvision.transforms.RandomCrop(size=32, padding=4))
    if random_resized_crop:
        transforms['train'].append(torchvision.transforms.RandomResizedCrop(size=32, scale=(0.75, 1.), ratio=(1., 1.)))
    for t in ['train', 'test']:
        transforms[t].append(torchvision.transforms.ToTensor())
    if random_erasing:
        transforms['train'].append(torchvision.transforms.RandomErasing())
    if normalize_to_plus_minus_one or normalize_to_unit_gaussian:
        # For the different normalization values see:
        # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/7
        if normalize_to_unit_gaussian:
            # These normalization values are taken from https://github.com/kuangliu/pytorch-cifar/issues/19
            # normalization_values = [(0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)]

            # These normalization values are taken from https://github.com/louity/patches
            # and also https://stackoverflow.com/questions/50710493/cifar-10-meaningless-normalization-values
            normalization_values = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
        else:
            normalization_values = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
        for t in ['train', 'test']:
            transforms[t].append(torchvision.transforms.Normalize(*normalization_values))

    datasets = {t: torchvision.datasets.CIFAR10(root='./data',
                                                train=(t == 'train'),
                                                transform=torchvision.transforms.Compose(transforms[t]),
                                                download=False)
                for t in ['train', 'test']}

    dataloaders = {t: torch.utils.data.DataLoader(datasets[t],
                                                  batch_size=batch_size,
                                                  shuffle=(t == 'train'),
                                                  num_workers=2)
                   for t in ['train', 'test']}

    return dataloaders


def get_model_device(model: Optional[torch.nn.Module]):
    """Returns the device of the given model
    """
    default_device = torch.device('cpu')

    # If the model is None, assume the model's device is CPU.
    if model is None:
        return default_device

    try:
        device = next(model.parameters()).device
    except StopIteration:  # If the model has no parameters, assume the model's device is CPU.
        device = default_device

    return device


def power_minus_1(a: torch.Tensor):
    """Raises the input tensor to the power of minus 1.
    """
    return torch.divide(torch.ones_like(a), a)


@torch.no_grad()
def get_model_output_shape(model: nn.Module, dataloader: Optional[DataLoader] = None):
    """Gets the output shape of the given model, on images from the given dataloader.
    """
    if dataloader is None:
        clean_dataloaders = get_dataloaders(batch_size=1)
        dataloader = clean_dataloaders["train"]

    inputs, _ = next(iter(dataloader))
    inputs = inputs.to(get_model_device(model))
    outputs = model(inputs)
    outputs = outputs.cpu().numpy()
    return outputs.shape[1:]  # Remove the first dimension corresponding to the batch


def get_whitening_matrix_from_covariance_matrix(covariance_matrix: np.ndarray,
                                                whitening_regularization_factor: float,
                                                zca_whitening: bool = False) -> np.ndarray:
    """Calculates the whitening matrix from the given covariance matrix.

    Args:
        covariance_matrix: The covariance matrix.
        whitening_regularization_factor: The regularization factor used when calculating the whitening,
            which is some small constant positive float added to the denominator.
        zca_whitening: Whether it's ZCA whitening (or PCA whitening).

    Returns:
        The whitening matrix.
    """
    eigenvectors, eigenvalues, eigenvectors_transposed = np.linalg.svd(covariance_matrix, hermitian=True)
    inv_sqrt_eigenvalues = np.diag(1. / (np.sqrt(eigenvalues) + whitening_regularization_factor))
    whitening_matrix = eigenvectors.dot(inv_sqrt_eigenvalues)
    if zca_whitening:
        whitening_matrix = whitening_matrix @ eigenvectors.T
    whitening_matrix = whitening_matrix.astype(np.float32)
    return whitening_matrix


def whiten_data(data, whitening_regularization_factor=1e-05, zca_whitening=False):
    """Whiten the given data.

    Note that the data is assumed to be of shape (n_samples, n_features), meaning it's a collection of row-vectors.

    Args:
        data: The given data to whiten.
        whitening_regularization_factor: The regularization factor used when calculating the whitening,
            which is some small constant positive float added to the denominator.
        zca_whitening: Whether it's ZCA whitening (or PCA whitening).

    Returns:
        The whitened data.
    """
    covariance_matrix = calc_covariance(data)
    whitening_matrix = get_whitening_matrix_from_covariance_matrix(covariance_matrix,
                                                                   whitening_regularization_factor,
                                                                   zca_whitening)
    centered_data = data - data.mean(axis=0)
    whitened_data = centered_data @ whitening_matrix
    return whitened_data


def normalize_data(data, epsilon=1e-05):
    """Normalize the given data (making it centered (zero mean) and each feature have unit variance).

    Note that the data is assumed to be of shape (n_samples, n_features), meaning it's a collection of row-vectors.

    Args:
        data: The data to normalize.
        epsilon: Some small positive number to add to the denominator,
            to avoid getting NANs (if the data-point has a small std).

    Returns:
        The normalized data.
    """
    centered_data = data - data.mean(axis=0)
    normalized_data = centered_data / (centered_data.std(axis=0) + epsilon)
    return normalized_data


def get_random_initialized_conv_kernel_and_bias(in_channels: int,
                                                out_channels: int,
                                                kernel_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns randomly initialized kernel and bias for a conv layer, as in PyTorch default initialization (Xavier).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: The kernel size.

    Returns:
        A tuple of two numpy arrays which are the randomly initialized kernel and bias for a conv layer,
        as in PyTorch default initialization (Xavier).
    """
    tmp_conv = nn.Conv2d(in_channels, out_channels, kernel_size)
    kernel = tmp_conv.weight.data.cpu().numpy().copy()
    bias = tmp_conv.bias.data.cpu().numpy().copy()
    return kernel, bias


@torch.no_grad()
def sample_random_patches(dataloader,
                          n_patches,
                          patch_size,
                          existing_model: Optional[nn.Module] = None,
                          visualize: bool = False,
                          random_uniform_patches: bool = False,
                          random_gaussian_patches: bool = False,
                          verbose: bool = False):
    """Sample random patches from the data.

    Args:
        dataloader: The dataloader to sample patches from.
        n_patches: Number of patches to sample.
        patch_size: The size of the patches to sample.
        existing_model: (Possibly) an existing model to transform each image from the dataloader.
        visualize: Whether to visualize the sampled patches (for debugging purposes).
        random_uniform_patches: Whether to avoid sampling and simply return patches from the uniform distribution.
        random_gaussian_patches: Whether to avoid sampling and simply return patches from the Gaussian distribution.
        verbose: Whether to print progress using tqdm.

    Returns:
        The sampled patches as a NumPy array.
    """
    batch_size = dataloader.batch_size
    n_images = len(dataloader.dataset)

    # We need the shape of the images in the data.
    # In relatively small datasets (CIFAR, MNIST) the data itself is stored in `dataloader.dataset.data`
    # and in ImageNet it's not the case since the data is too large.
    # This is why the shape of ImageNet images is hard-coded.
    images_shape = dataloader.dataset.data.shape[1:] if hasattr(dataloader.dataset, 'data') else (224, 224, 3)

    if len(images_shape) == 2:  # When the dataset contains grayscale images,
        images_shape += (1,)  # add dimension of channels which will be 1.

    images_shape = np.roll(images_shape, shift=1)  # In the dataset it's H x W x C but in the model it's C x H x W
    if existing_model is not None:
        device = get_model_device(existing_model)
        images_shape = get_model_output_shape(existing_model, dataloader)

    if len(images_shape) > 1:
        assert len(images_shape) == 3 and (images_shape[1] == images_shape[2]), "Should be C x H x W where H = W"
        spatial_size = images_shape[-1]
        if patch_size == -1:  # -1 means the patch size is the whole size of the image.
            patch_size = spatial_size
        n_patches_per_row_or_col = spatial_size - patch_size + 1
        patch_shape = (images_shape[0],) + 2 * (patch_size,)
    else:
        assert patch_size == -1, "When working with fully-connected the patch 'size' must be -1 i.e. the whole size."
        n_patches_per_row_or_col = 1
        patch_shape = images_shape

    n_patches_per_image = n_patches_per_row_or_col ** 2
    n_patches_in_dataset = n_images * n_patches_per_image

    if n_patches >= n_patches_in_dataset:
        n_patches = n_patches_in_dataset

    patches_indices_in_dataset = np.random.default_rng().choice(n_patches_in_dataset, size=n_patches, replace=False)

    images_indices = patches_indices_in_dataset % n_images
    patches_indices_in_images = patches_indices_in_dataset // n_images
    patches_x_indices_in_images = patches_indices_in_images % n_patches_per_row_or_col
    patches_y_indices_in_images = patches_indices_in_images // n_patches_per_row_or_col

    batches_indices = images_indices // batch_size
    images_indices_in_batches = images_indices % batch_size

    patches = np.empty(shape=(n_patches,) + patch_shape, dtype=np.float32)

    if random_uniform_patches:
        return np.random.default_rng().uniform(low=-1, high=+1, size=patches.shape).astype(np.float32)
    if random_gaussian_patches:
        patch_dim = math.prod(patch_shape)
        return np.random.default_rng().multivariate_normal(
            mean=np.zeros(patch_dim), cov=np.eye(patch_dim), size=n_patches).astype(np.float32).reshape(patches.shape)

    iterator = enumerate(dataloader)
    if verbose:
        iterator = tqdm(iterator, total=len(dataloader), desc='Sampling patches from the dataset')
    for batch_index, (inputs, _) in iterator:
        if batch_index not in batches_indices:
            continue

        relevant_patches_mask = (batch_index == batches_indices)
        relevant_patches_indices = np.where(relevant_patches_mask)[0]

        if existing_model is not None:
            inputs = inputs.to(device)
            inputs = existing_model(inputs)
        inputs = inputs.cpu().numpy()

        for i in relevant_patches_indices:
            image_index_in_batch = images_indices_in_batches[i]
            if len(patch_shape) > 1:
                patch_x_start = patches_x_indices_in_images[i]
                patch_y_start = patches_y_indices_in_images[i]
                patch_x_slice = slice(patch_x_start, patch_x_start + patch_size)
                patch_y_slice = slice(patch_y_start, patch_y_start + patch_size)

                patches[i] = inputs[image_index_in_batch, :, patch_x_slice, patch_y_slice]

                if visualize:
                    visualize_image_patch_pair(image=inputs[image_index_in_batch], patch=patches[i],
                                               patch_x_start=patch_x_start, patch_y_start=patch_y_start)
            else:
                patches[i] = inputs[image_index_in_batch]

    return patches


def visualize_image_patch_pair(image, patch, patch_x_start, patch_y_start):
    """Visualize the given image and the patch in it, with rectangle in the location of the patch.
    """
    patch_size = patch.shape[-1]
    rect = Rectangle(xy=(patch_y_start, patch_x_start),  # x and y are reversed on purpose...
                     width=patch_size, height=patch_size,
                     linewidth=1, edgecolor='red', facecolor='none')

    plt.figure()
    ax = plt.subplot(2, 1, 1)
    ax.imshow(np.transpose(image, axes=(1, 2, 0)))
    ax.add_patch(rect)
    ax = plt.subplot(2, 1, 2)
    ax.imshow(np.transpose(patch, axes=(1, 2, 0)))
    plt.show()


def cross_entropy_gradient(logits, labels):
    """
    Calculate the gradient of the cross-entropy loss with respect to the input logits.
    Note the cross-entropy loss in PyTorch basically calculates log-softmax followed by negative log-likelihood loss.
    Therefore, the gradient is the softmax output of the logits, where in the labels indices a 1 is subtracted.

    Inspiration from http://machinelearningmechanic.com/deep_learning/2019/09/04/cross-entropy-loss-derivative.html

    :param logits: The raw scores which are the input to the cross-entropy-loss.
    :param labels: The labels (for each i the index of the true class of this training-sample).
    :return: The gradient of the cross-entropy loss.
    """
    # This is the probabilities vector obtained using the softmax function on the raw scores.
    p = torch.nn.functional.softmax(logits, dim=1)

    # Subtract 1 from the labels indices, which gives the final gradient of the cross-entropy loss.
    p.scatter_add_(dim=1, index=labels.unsqueeze(dim=-1), src=torch.full_like(p, fill_value=-1))

    return p


def evaluate_model_with_last_gradient(model, criterion, dataloader, device, training_step=None,
                                      log_to_wandb: bool = True):
    """
    Evaluate the given model on the test set.
    In addition to returning the final test loss & accuracy,
    this function evaluate each one of the model local modules (by logging to wandb).

    :param model: The model
    :param criterion: The criterion.
    :param dataloader: The test set data-loader.
    :param device: The device to use.
    :param training_step: The training-step (integer), important to wandb logging.
    :param log_to_wandb: Whether to log to wandb or not.
    :return: The test set loss and accuracy.
    """
    raise DeprecationWarning('This function is deprecated, and will be removed in the future.')

    model.eval()

    modules_accumulators = [Accumulator() if (aux_net is not None) else None for aux_net in model.auxiliary_nets]

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            aux_nets_outputs = model(inputs)

            aux_nets_losses = [criterion(outputs, labels) if (outputs is not None) else None
                               for outputs in aux_nets_outputs]
            aux_nets_predictions = [torch.max(outputs, dim=1)[1] if (outputs is not None) else None
                                    for outputs in aux_nets_outputs]

            # Update the corresponding accumulators to visualize the performance of each module.
            for i in range(len(model.blocks)):
                if model.auxiliary_nets[i] is not None:
                    modules_accumulators[i].update(
                        mean_loss=aux_nets_losses[i].item(),
                        num_corrects=torch.sum(torch.eq(aux_nets_predictions[i], labels.data)).item(),
                        n_samples=inputs.size(0)
                    )

    if log_to_wandb:
        assert training_step is not None
        for i, modules_accumulator in enumerate(modules_accumulators):
            if modules_accumulator is not None:
                wandb.log(data=modules_accumulator.get_dict(prefix=f'module#{i}_test'), step=training_step)

    final_accumulator = modules_accumulators[-2]  # Last one is None because last block is MaxPool with no aux-net.
    return final_accumulator.get_mean_loss(), final_accumulator.get_accuracy()


def evaluate_local_model(model, criterion, dataloader, device, training_step=None, log_to_wandb: bool = True):
    """
    Evaluate the given model on the test set.
    In addition to returning the final test loss & accuracy,
    this function evaluate each one of the model local modules (by logging to wandb).

    :param model: The model
    :param criterion: The criterion.
    :param dataloader: The test set data-loader.
    :param device: The device to use.
    :param training_step: The training-step (integer), important to wandb logging.
    :param log_to_wandb: Whether to log to wandb or not.
    :return: The test set loss and accuracy.
    """
    raise DeprecationWarning('This function is deprecated, and will be removed in the future.')

    model.eval()
    n_modules = len(model.blocks)

    modules_accumulators = [Accumulator() if (aux_net is not None) else None for aux_net in model.auxiliary_nets]

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            inputs_representation = inputs
            for i in range(n_modules):
                result = model(inputs_representation, first_block_index=i, last_block_index=i)
                inputs_representation, outputs = result[0], result[1]
                if outputs is not None:
                    loss = criterion(outputs, labels)

                    modules_accumulators[i].update(
                        mean_loss=loss.item(),
                        num_corrects=torch.sum(torch.eq(torch.max(outputs, dim=1)[1], labels.data)).item(),
                        n_samples=inputs.size(0)
                    )

    if log_to_wandb:
        assert training_step is not None
        for i, modules_accumulator in enumerate(modules_accumulators):
            if modules_accumulator is not None:
                wandb.log(data=modules_accumulator.get_dict(prefix=f'module#{i}_test'), step=training_step)

    final_accumulator = modules_accumulators[-2]  # Last one is None because last block is MaxPool with no aux-net.
    return final_accumulator.get_mean_loss(), final_accumulator.get_accuracy()


def evaluate_model(model, criterion, dataloader, device, inputs_preprocessing_function=None):
    """
    Evaluate the given model on the test set.

    :param model: The model
    :param criterion: The criterion.
    :param dataloader: The test set data-loader.
    :param device: The device to use.
    :return: The test set loss and accuracy.
    """
    raise DeprecationWarning('This function is deprecated, and will be removed in the future.')

    model.eval()

    loss_sum = 0.0
    corrects_sum = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            if inputs_preprocessing_function is not None:
                inputs = inputs_preprocessing_function(inputs)
            result = model(inputs)

            # In DGL the model forward function also return the inputs representation
            # (in addition to the classes' scores which are the prediction of the relevant auxiliary network)
            outputs = result[1] if isinstance(result, tuple) else result
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        loss_sum += loss.item() * inputs.size(0)
        corrects_sum += torch.sum(torch.eq(predictions, labels.data)).item()

    loss = loss_sum / len(dataloader.dataset)
    accuracy = 100 * (corrects_sum / len(dataloader.dataset))

    return loss, accuracy


def perform_train_step_dgl(model, inputs, labels, criterion, optimizers, training_step,
                           modules_accumulators,
                           log_interval: int = 100):
    """
    Perform a train-step for a model trained with DGL.
    The difference between the regular train-step and this one is that the model forward pass
    is done iteratively for each block in the model, performing backward pass and optimizer step for each block
    (using its corresponding auxiliary network).

    :param model: The model.
    :param inputs: The inputs.
    :param labels: The labels.
    :param criterion: The criterion.
    :param optimizers: The optimizers (one for each local module in the whole model).
    :param training_step: The training-step (integer), important to wandb logging.
    :param modules_accumulators: Accumulators for each local module.
    :param log_interval: How many training/testing steps between each logging (to wandb).
    :return: The loss of this train-step, as well as the predictions.
    """
    raise DeprecationWarning('This function is deprecated, and will be removed in the future.')

    inputs_representation = torch.clone(inputs)
    loss, predictions = None, None

    for i in range(len(model.blocks)):
        inputs_representation, outputs = model(inputs_representation, first_block_index=i, last_block_index=i)
        if outputs is not None:
            assert optimizers[i] is not None, "If the module has outputs it means it has an auxiliary-network " \
                                              "attached so it should has trainable parameters to optimize."
            predictions = torch.max(outputs, dim=1)[1]  # Save to return the predictions (argmax) later.
            optimizers[i].zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizers[i].step()

            modules_accumulators[i].update(
                mean_loss=loss.item(),
                num_corrects=torch.sum(torch.eq(predictions, labels.data)).item(),
                n_samples=inputs.size(0)
            )

        inputs_representation = inputs_representation.detach()  # Prepare the input tensor for the next block.

    if training_step % log_interval == 0:
        for i, modules_accumulator in enumerate(modules_accumulators):
            if modules_accumulator is not None:
                wandb.log(data=modules_accumulator.get_dict(prefix=f'module#{i}_train'), step=training_step)
                modules_accumulator.reset()

    return loss.item(), predictions


def perform_train_step_ssl(model, inputs, labels, scores_criterion, optimizers, training_step,
                           ssl_criterion=None,
                           pred_loss_weight: float = 1, ssl_loss_weight: float = 0.1,
                           first_trainable_block: int = 0,
                           shift_ssl_labels: bool = False,
                           images_log_interval: int = 1000):
    """
    Perform a train-step for a model trained with local self-supervised loss, possibly in combination with DGL.

    :param model: The model.
    :param inputs: The inputs.
    :param labels: The labels.
    :param scores_criterion: The criterion.
    :param optimizers: The optimizers.
    :param training_step: The training-step (integer), important to wandb logging.
    :param ssl_criterion: The loss for the SSL outputs (regression, e.g. L1 or L2).
    :param pred_loss_weight: When combining with DGL, the weight for the scores' prediction loss.
    :param ssl_loss_weight: When combining with DGL, the weight for the SSL prediction loss.
    :param first_trainable_block: Can be used to freeze a few layers in their initial weights.
    :param shift_ssl_labels: Whether to shift the SSL labels or keep them the original images.
    :param images_log_interval: Frequency of logging images to wandb
                                (less frequent then regular logging because it's quite heavy).
    :return: The loss of this train-step, as well as the predictions.
    """
    raise DeprecationWarning('This function is deprecated, and will be removed in the future.')

    inputs_size = inputs.size(-1)
    inputs_representation = torch.clone(inputs)
    loss, predictions, scores_loss, ssl_loss = None, None, None, None
    n_plot_images = 4
    indices_to_plot = np.random.choice(inputs.size(0), size=n_plot_images, replace=False)
    shifts = tuple(np.linspace(start=5, stop=16, num=len(model.blocks), dtype=int))

    if first_trainable_block < 0:
        first_trainable_block = len(model.blocks) + first_trainable_block
    assert 0 <= first_trainable_block < len(model.blocks), f"Invalid first_trainable_block ({first_trainable_block})."

    for i in range(len(model.blocks)):
        inputs_representation, scores, ssl_outputs = model(inputs_representation,
                                                           first_block_index=i, last_block_index=i)
        if i < first_trainable_block:
            continue

        if scores is not None:
            assert ssl_outputs is not None
            assert optimizers[i] is not None, "If the module has outputs it means it has an auxiliary-network " \
                                              "attached so it should has trainable parameters to optimize."
            predictions = torch.max(scores, dim=1)[1]
            optimizers[i].zero_grad()

            # Note that in the last block we always want to punish according to the predicted classes' scores,
            # nd ignore SSL. Since the block in all of the models is simply MaxPool layer,
            # by 'last block' we mean one before the last.
            if i == len(model.blocks) - 2:
                loss = scores_criterion(scores, labels)
            else:
                ssl_outputs_size = ssl_outputs.size(-1)
                ssl_labels = inputs if ssl_outputs_size == inputs_size else resize(inputs, size=[ssl_outputs_size] * 2)
                if shift_ssl_labels:
                    ssl_labels = torch.roll(ssl_labels, shifts=(shifts[i], shifts[i]), dims=(2, 3))

                ssl_loss = ssl_criterion(ssl_outputs, ssl_labels)

                if pred_loss_weight > 0:
                    scores_loss = scores_criterion(scores, labels)
                    loss = pred_loss_weight * scores_loss + ssl_loss_weight * ssl_loss
                else:
                    loss = ssl_loss

                if training_step % images_log_interval == 0:
                    images = torch.cat([ssl_labels[indices_to_plot], ssl_outputs[indices_to_plot].detach()])
                    grid = torchvision.utils.make_grid(images, nrow=n_plot_images)
                    wandb_image = wandb.Image(grid.cpu().numpy().transpose((1, 2, 0)))
                    wandb.log({f'SSL-layer#{i}': [wandb_image]}, step=training_step)

            loss.backward()
            optimizers[i].step()
        else:
            assert ssl_outputs is None

        inputs_representation = inputs_representation.detach()  # Prepare the input tensor for the next block.

    return loss.item(), predictions, scores_loss, ssl_loss


def perform_train_step_direct_global(model, inputs, labels, criterion, optimizers,
                                     training_step, modules_accumulators, last_gradient_weight: float = 0.5,
                                     log_interval: int = 100):
    """
    Perform a train-step for a model with "direct-global-feedback".

    :param model: The model.
    :param inputs: The inputs.
    :param labels: The labels.
    :param criterion: The criterion.
    :param optimizers: The optimizers.
    :param training_step: The training-step (integer), important to wandb logging.
    :param modules_accumulators: Accumulators for each local module.
    :param last_gradient_weight: Weight of the last gradient in each intermediate gradient calculator.
    :param log_interval: How many training/testing steps between each logging (to wandb).
    :return: The loss of this train-step, as well as the predictions.
    """
    raise DeprecationWarning('This function is deprecated, and will be removed in the future.')

    inputs_representation = torch.clone(inputs)
    loss_value, predictions = None, None

    modules_outputs: List[torch.Tensor] = list()

    # Perform the forward-pass.
    for i in range(len(model.blocks)):
        inputs_representation, outputs = model(inputs_representation, first_block_index=i, last_block_index=i)
        modules_outputs.append(outputs)

        if i == len(model.blocks) - 2:
            predictions = torch.max(outputs, dim=1)[1]

        inputs_representation = inputs_representation.detach()  # Prepare the input tensor for the next block.

    # Can't change variables within inner functions, but can change inner state of mutable variables.
    # https://stackoverflow.com/questions/11987358/
    last_layer_grad = dict(value=None)  # This will hold the gradient of the last layer.

    def last_module_hook(grad):
        assert last_layer_grad['value'] is None, "\'last_layer_grad\' should not have been set yet."
        last_layer_grad['value'] = grad

    def intermediate_module_hook(grad):
        assert last_layer_grad['value'] is not None, "\'last_layer_grad\' should have been set."
        return (1 - last_gradient_weight) * grad + last_gradient_weight * last_layer_grad['value']

    # Perform the backward-pass, using the gradients of the last layer w.r.t the outputs tensor.
    for i in range(len(model.blocks) - 1, -1, -1):
        module_outputs = modules_outputs[i]
        module_optimizer = optimizers[i]

        if module_outputs is not None:
            assert module_optimizer is not None
            module_optimizer.zero_grad()
            loss = criterion(module_outputs, labels)

            modules_accumulators[i].update(
                mean_loss=loss.item(),
                num_corrects=torch.sum(torch.eq(torch.max(module_outputs, dim=1)[1], labels.data)).item(),
                n_samples=inputs.size(0)
            )

            if last_layer_grad['value'] is None:
                assert i == len(model.blocks) - 2, "This should happen in the last block (that is not MaxPool)."
                module_outputs.register_hook(last_module_hook)
                loss_value = loss.item()  # This loss will be returned eventually, since it's the final model's loss.
            else:
                module_outputs.register_hook(intermediate_module_hook)

            loss.backward()
            module_optimizer.step()

    if training_step % log_interval == 0:
        for i, modules_accumulator in enumerate(modules_accumulators):
            if modules_accumulator is not None:
                wandb.log(data=modules_accumulator.get_dict(prefix=f'module#{i}_train'), step=training_step)
                modules_accumulator.reset()

    return loss_value, predictions


def bdot(a, b):
    """
    Calculates batch-wise dot-product, as in:
    https://github.com/pytorch/pytorch/issues/18027#issuecomment-473119948
    """
    batch_size = a.shape[0]
    dimension = a.shape[1]
    return torch.bmm(a.view(batch_size, 1, dimension),
                     b.view(batch_size, dimension, 1)).reshape(-1)


def perform_train_step_last_gradient(model, inputs, labels, criterion, optimizer,
                                     training_step, modules_accumulators,
                                     last_gradient_weight: float = 0.5, log_interval: int = 100):
    """
    Perform a train-step for a model trained with the last gradient in each intermediate module.

    :param model: The model.
    :param inputs: The inputs.
    :param labels: The labels.
    :param criterion: The criterion.
    :param optimizer: The optimizer.
    :param training_step: The training-step (integer), important to wandb logging.
    :param modules_accumulators: Accumulators for each local module.
    :param last_gradient_weight: Weight of the last gradient in each intermediate gradient calculator.
    :param log_interval: How many training/testing steps between each logging (to wandb).
    :return: The loss of this train-step, as well as the predictions.
    """
    raise DeprecationWarning('This function is deprecated, and will be removed in the future.')

    optimizer.zero_grad()
    minibatch_size = inputs.size(0)

    aux_nets_outputs = model(inputs)

    aux_nets_losses = [criterion(outputs, labels) if (outputs is not None) else None
                       for outputs in aux_nets_outputs]
    aux_nets_predictions = [torch.max(outputs, dim=1)[1] if (outputs is not None) else None
                            for outputs in aux_nets_outputs]

    last_logits = aux_nets_outputs[-2]  # minus 2 because the last block is a pooling layer
    last_loss = aux_nets_losses[-2]
    last_gradient = (1 / minibatch_size) * cross_entropy_gradient(last_logits, labels).detach()

    dummy_losses = [torch.mean(bdot(last_gradient, aux_net_outputs))
                    for aux_net_outputs in aux_nets_outputs[:-2]
                    if aux_net_outputs is not None]

    loss = (last_loss +
            (1 - last_gradient_weight) * torch.sum(torch.stack([l for l in aux_nets_losses if l is not None])) +
            last_gradient_weight * torch.sum(torch.stack(dummy_losses)))

    # This line (instead of the above loss definition) gives DGL equivalent implementation.
    # loss = torch.sum(torch.stack([l for l in aux_nets_losses if l is not None]))

    loss.backward()
    optimizer.step()

    # Update the corresponding accumulators to visualize the performance of each module.
    for i in range(len(model.blocks)):
        if model.auxiliary_nets[i] is not None:
            modules_accumulators[i].update(
                mean_loss=aux_nets_losses[i].item(),
                num_corrects=torch.sum(torch.eq(aux_nets_predictions[i], labels.data)).item(),
                n_samples=minibatch_size
            )

    # Visualize the performance of each module once in a while.
    if training_step % log_interval == 0:
        for i, modules_accumulator in enumerate(modules_accumulators):
            if modules_accumulator is not None:
                wandb.log(data=modules_accumulator.get_dict(prefix=f'module#{i}_train'), step=training_step)
                modules_accumulator.reset()

    return aux_nets_losses[-2].item(), aux_nets_predictions[-2]


def perform_train_step(model, inputs, labels, criterion, optim,
                       training_step, is_dgl, ssl, ssl_criterion,
                       pred_loss_weight, ssl_loss_weight, first_trainable_block, shift_ssl_labels,
                       is_direct_global, modules_accumulators, last_gradient_weight, use_last_gradient):
    """
    Perform a single train-step, which is done differently when using regular training, DGL and cDNI.

    :param model: The model.
    :param inputs: The inputs.
    :param labels: The labels.
    :param criterion: The criterion.
    :param optim: The optimizer (or plural optimizers).
    :param training_step: The training-step (integer), important to wandb logging.
    :param is_dgl: Whether this model is trained using DGL (affects the optimizer and the train-step functionality).
    :param ssl: Whether this model is trained using SSL.
    :param ssl_criterion: The criterion for the SSL predictions.
    :param pred_loss_weight: When combining with DGL, the weight for the scores' prediction loss.
    :param ssl_loss_weight: When combining with DGL, the weight for the SSL prediction loss.
    :param first_trainable_block: Can be used to freeze a few layers in their initial weights.
    :param shift_ssl_labels: Whether to shift the SSL labels or keep them the original images.
    :param is_direct_global: Whether to use direct global gradient or not.
    :param modules_accumulators: Accumulators for each local module.
    :param last_gradient_weight: Weight of the last gradient in each intermediate gradient calculator.
    :return: The loss of this train-step, as well as the predictions.
    """
    raise DeprecationWarning('This function is deprecated, and will be removed in the future.')

    mutual_args = (model, inputs, labels, criterion, optim)

    if is_dgl:
        if is_direct_global:
            return perform_train_step_direct_global(*mutual_args, training_step, modules_accumulators,
                                                    last_gradient_weight)
        elif use_last_gradient:
            return perform_train_step_last_gradient(*mutual_args, training_step, modules_accumulators,
                                                    last_gradient_weight)
        elif ssl:
            return perform_train_step_ssl(*mutual_args, training_step, ssl_criterion, pred_loss_weight, ssl_loss_weight,
                                          first_trainable_block, shift_ssl_labels)
        else:
            return perform_train_step_dgl(*mutual_args, training_step, modules_accumulators)
    else:
        return perform_train_step_regular(*mutual_args)


def get_args_from_flattened_dict(args_class, flattened_dict, excluded_categories: Optional[List[str]] = None):
    args_dict = dict()

    for arg_name, value in flattened_dict.items():
        categories = list(args_class.__fields__.keys())
        found = False
        for category in categories:
            category_args = list(args_class.__fields__[category].default.__fields__.keys())
            if arg_name in category_args:
                if category not in args_dict:
                    args_dict[category] = dict()
                args_dict[category][arg_name] = value
                found = True
        if not found:
            raise ValueError(f'Argument {arg_name} is not recognized.')

    if excluded_categories is not None:
        for category in excluded_categories:
            args_dict.pop(category)

    args = args_class.parse_obj(args_dict)
    return args


class ShuffleTensor(nn.Module):
    def __init__(self,
                 spatial_size: int = 32,
                 channels: int = 3,
                 spatial_only: bool = True,
                 fixed_permutation: bool = True):
        """A data transformation which shuffles the pixels of the input image.

        Args:
            spatial_size: The spatial size of the input tensor (it's assumed to be square so height=width=spatial_size).
            channels: The number of channels of the input tensor.
            spatial_only: If it's true, shuffle the spatial locations only and the channels dimension will stay intact.
            fixed_permutation: If it's true, a fixed permutation will be used every time this module is called.
        """
        super().__init__()
        self.spatial_size = spatial_size
        self.channels = channels
        self.spatial_only = spatial_only

        permutation_size = self.spatial_size ** 2
        if (not self.spatial_only) or (self.spatial_size == 1):
            # The 1st term in the `if` means we are shuffling a convolutional layer including the channels dimension,
            # and the 2nd term in the `if` means we are shuffling a fully-connected layer.
            # Anyway, we need to include the channels dimension in the permutation.
            permutation_size *= self.channels
        self.permutation = torch.randperm(permutation_size) if fixed_permutation else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim >= 3:
            # The number of dimensions can be 4 (batched tensors) or 3 (a single tensor, e.g. when used as data
            # preprocessing function a.k.a. `transform`). Anyway, last three dimensions are C x H x W.
            assert x.shape[-1] == x.shape[-2] == self.spatial_size, f'{x.shape=} ; {self.spatial_size=}'
            assert x.shape[-3] == self.channels, f'{x.shape=} ; {self.channels=}'
            start_dim = -2 if self.spatial_only else -3
        elif x.ndim == 2:
            assert self.spatial_size == 1
            assert x.shape[-1] == self.channels, f'{x.shape=} ; {self.channels=}'
            start_dim = -1  # Will cause flatten() later to have not effect, since the data is already flattened.
        else:
            assert False, f'x should not be 1-dimensional - {x.shape=}'

        x_flat = torch.flatten(x, start_dim=start_dim)
        permutation = torch.randperm(x_flat.shape[-1]) if (self.permutation is None) else self.permutation
        permuted_x_flat = x_flat[..., permutation]
        permuted_x = torch.reshape(permuted_x_flat, shape=x.shape)
        return permuted_x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)  # When used as a data transform it's required to be "callable".

    def extra_repr(self) -> str:
        return f'spatial_size={self.spatial_size}; ' \
               f'channels={self.channels}; ' \
               f'spatial_only={self.spatial_only}; ' \
               f'fixed_permutation={self.permutation is None}'


class ImageNet(ImageFolder):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root: str, split: Literal['train', 'val'] = 'train', **kwargs: Any) -> None:
        self.root = root
        self.split = split

        wnid_to_classes = torch.load(os.path.join(self.root, 'meta.bin'))[0]

        super(ImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


def run_kmeans_clustering(data: np.ndarray, k: int, try_to_use_faiss: bool):
    """Runs k-means clustering on the given data. Returns the centroids, the assignments of data-points to centroids,
    and the distances between each data-point to its assigned centroid.

    Args:
        data: ndarray of shape (n_samples, n_features)
            The data to cluster.
        k: int
            The number of clusters to form as well as the number of centroids to generate.
        try_to_use_faiss: boolean, default=False
            Whether to use faiss library for faster run-time (requires faiss library installed).

    Returns:
        centroids: ndarray of shape (n_clusters, n_features)
            The clusters centers.
        indices: ndarray of shape (n_samples,)
            Labels of each point (i.e., the index of the closest centroid).
        distances: ndarray of shape (n_samples,)
            The distance of each data-point to its closest centroid.
    """
    if try_to_use_faiss:
        try:
            from faiss import Kmeans as KMeans
            kmeans = KMeans(d=data.shape[1], k=k)
            kmeans.train(data)
            centroids = kmeans.centroids
            distances, indices = kmeans.assign(data)
            return centroids, distances, indices
        except ImportError:
            warnings.warn(f'use_faiss is True, but failed to import faiss. Using sklearn instead.')
            from sklearn.cluster import KMeans

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k)
    distances_to_each_centroid = kmeans.fit_transform(data)
    indices = kmeans.labels_
    distances = np.take_along_axis(distances_to_each_centroid, indices[:, np.newaxis], axis=1).squeeze(axis=1)
    centroids = kmeans.cluster_centers_
    return centroids, indices, distances


class DataModule(LightningDataModule):
    def __init__(self, args: DataArgs, batch_size: int):
        """A datamodule to be used with PyTorch Lightning modules.

        Args:
            args: The data's arguments-schema.
            batch_size: The batch-size.
        """
        super().__init__()

        self.dataset_class = self.get_dataset_class(args.dataset_name)
        self.n_channels = args.n_channels
        self.spatial_size = args.spatial_size
        self.data_dir = args.data_dir
        self.num_workers = args.num_workers
        self.batch_size = batch_size

        transforms_list_no_aug, transforms_list_with_aug = self.get_transforms_lists(args)
        transforms_clean = [ToTensor()]
        if self.dataset_class is ImageNet:  # Since ImageNet images are of sifferent sizes, we must resize
            transforms_clean = [Resize((args.spatial_size, args.spatial_size))] + transforms_clean
        self.transforms = {'aug': Compose(transforms_list_with_aug),
                           'no_aug': Compose(transforms_list_no_aug),
                           'clean': Compose(transforms_clean)}
        self.datasets = {f'{stage}_{aug}': None
                         for stage in ('fit', 'validate')
                         for aug in ('aug', 'no_aug', 'clean')}

    def get_dataset_class(self, dataset_name: str) -> Type[Union[CIFAR10, CIFAR100, MNIST, FashionMNIST, ImageNet]]:
        """Gets the class of the dataset, according to the given dataset name.

        Args:
            dataset_name: name of the dataset (CIFAR10, CIFAR100, MNIST, FashionMNIST or ImageNet).

        Returns:
            The dataset class.
        """
        if dataset_name == 'CIFAR10':
            return CIFAR10
        elif dataset_name == 'CIFAR100':
            return CIFAR100
        elif dataset_name == 'MNIST':
            return MNIST
        elif dataset_name == 'FashionMNIST':
            return FashionMNIST
        elif dataset_name == 'ImageNet':
            return ImageNet
        else:
            raise NotImplementedError(f'Dataset {dataset_name} is not implemented.')

    def get_transforms_lists(self, args: DataArgs) -> Tuple[list, list]:
        """Gets the transformations list to be used in the dataloader.

        Args:
            args: The data's arguments-schema.

        Returns:
            One list is the transformations without augmentation,
            and the other is the transformations with augmentations.
        """
        augmentations = self.get_augmentations_transforms(args.random_horizontal_flip, args.random_crop)
        normalization = self.get_normalization_transform(args.normalization_to_plus_minus_one,
                                                         args.normalization_to_unit_gaussian)
        normalizations_list = list() if (normalization is None) else [normalization]
        pre_transforms = [ToTensor()]
        post_transforms = list() if not args.shuffle_images else [
            ShuffleTensor(args.spatial_size, args.n_channels, args.keep_rgb_triplets_intact, args.fixed_permutation)
        ]
        transforms_list_no_aug = pre_transforms + normalizations_list + post_transforms
        if self.dataset_class is ImageNet:  # Since ImageNet images are of sifferent sizes, we must resize
            transforms_list_no_aug = [Resize((args.spatial_size, args.spatial_size))] + transforms_list_no_aug

        transforms_list_with_aug = augmentations + pre_transforms + normalizations_list + post_transforms

        return transforms_list_no_aug, transforms_list_with_aug

    def get_augmentations_transforms(self, random_flip: bool, random_crop: bool) -> list:
        """Gets the augmentations transformations list to be used in the dataloader.

        Args:
            random_flip: Whether to use random-flip augmentation.
            random_crop: Whether to use random-crop augmentation.
                In ImageNet dataset, the layer that is being used is RandomResizedCrop
                and not padding followed by RandomCrop.
        Returns:
            A list containing the augmentations transformations.
        """
        augmentations_transforms = list()

        if self.dataset_class is ImageNet:
            if random_crop:
                augmentations_transforms.append(RandomResizedCrop(size=self.spatial_size))
            else:
                augmentations_transforms.append(Resize((self.spatial_size, self.spatial_size)))
        elif random_crop:
            augmentations_transforms.append(RandomCrop(size=self.spatial_size, padding=4))
        if random_flip:
            augmentations_transforms.append(RandomHorizontalFlip())

        return augmentations_transforms

    def get_normalization_transform(self, plus_minus_one: bool, unit_gaussian: bool) -> Optional[Normalize]:
        """Gets the normalization transformation to be used in the dataloader (or None, if no normalization is needed).

        Args:
            plus_minus_one: Whether to normalize the input-images from [0,1] to [-1,+1].
            unit_gaussian: Whether to normalize the input-images to have zero mean and std one (channels-wise).

        Returns:
            The normalization transformation (or None, if no normalization is needed).
        """
        assert not (plus_minus_one and unit_gaussian), 'Only one should be given'

        if unit_gaussian:
            if self.dataset_class is ImageNet:
                normalization_values = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
            elif self.dataset_class is CIFAR10:
                normalization_values = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
            elif self.dataset_class is CIFAR100:
                normalization_values = [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)]
            else:
                raise NotImplementedError('Normalization using mean and std is supported only for '
                                          'CIFAR10 / CIFAR100 / ImageNet.')
        elif plus_minus_one:
            normalization_values = [(0.5,) * self.n_channels] * 2  # times 2 because one is mean and one is std
        else:
            return None

        return Normalize(*normalization_values)

    def prepare_data(self):
        """Download the dataset if it's not already in `self.data_dir`.
        """
        if self.dataset_class is not ImageNet:  # ImageNet should be downloaded manually beforehand
            for train_mode in [True, False]:
                self.dataset_class(self.data_dir, train=train_mode, download=True)

    def setup(self, stage: Optional[str] = None):
        """Create the different datasets.
        """
        if stage is None:
            return

        for s in ('fit', 'validate'):
            for aug in ('aug', 'no_aug', 'clean'):
                k = f'{s}_{aug}'
                if self.dataset_class is ImageNet:
                    kwargs = dict(split='train' if s == 'fit' else 'val')
                else:
                    kwargs = dict(train=(s == 'fit'))
                if self.datasets[k] is None:
                    self.datasets[k] = self.dataset_class(self.data_dir,
                                                          transform=self.transforms[aug],
                                                          **kwargs)

    def train_dataloader(self):
        """
        Returns:
             The train dataloader, which is the train-data with augmentations.
        """
        return DataLoader(self.datasets['fit_aug'], batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True)

    def train_dataloader_no_aug(self):
        """
        Returns:
             The train dataloader without augmentations.
        """
        return DataLoader(self.datasets['fit_no_aug'], batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True)

    def train_dataloader_clean(self):
        """
        Returns:
             The train dataloader without augmentations and normalizations (i.e. the original images in [0,1]).
        """
        return DataLoader(self.datasets['fit_clean'], batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        """
        Returns:
             The validation dataloader, which is the validation-data without augmentations
             (but possibly has normalization, if the training-dataloader has one).
        """
        return DataLoader(self.datasets['validate_no_aug'], batch_size=self.batch_size, num_workers=self.num_workers)


def initialize_datamodule(args: DataArgs, batch_size: int):
    datamodule = DataModule(args, batch_size)
    datamodule.prepare_data()
    datamodule.setup(stage='fit')
    datamodule.setup(stage='validate')

    return datamodule


def initialize_wandb_logger(args, name_suffix: str = ''):
    run_name = None if (args.env.wandb_run_name is None) else args.env.wandb_run_name + name_suffix
    return WandbLogger(project='thesis', config=args.flattened_dict(), name=run_name, log_model=True)


def initialize_model(arch_args: ArchitectureArgs,
                     opt_args: OptimizationArgs,
                     data_args: DataArgs,
                     wandb_logger: WandbLogger):
    if arch_args.model_name.startswith('VGG'):
        model_class = LitVGG
    elif any(arch_args.model_name.startswith(s) for s in ['D-', 'S-']):
        model_class = LitCNN
    else:
        model_class = LitMLP

    if arch_args.use_pretrained:
        artifact = wandb_logger.experiment.use_artifact(arch_args.pretrained_path, type='model')
        artifact_dir = artifact.download()
        model = model_class.load_from_checkpoint(str(Path(artifact_dir) / "model.ckpt"),
                                                 arch_args=arch_args, opt_args=opt_args, data_args=data_args)
    else:
        model = model_class(arch_args, opt_args, data_args)

    return model


def initialize_trainer(env_args: EnvironmentArgs, opt_args: OptimizationArgs, wandb_logger: WandbLogger,
                       additional_callbacks: Optional[list] = None):
    trainer_kwargs = dict(logger=wandb_logger, max_epochs=opt_args.epochs,
                          enable_checkpointing=env_args.enable_checkpointing)
    callbacks = [ModelSummary(max_depth=5)]

    if isinstance(env_args.multi_gpu, list) or (env_args.multi_gpu != 0):
        trainer_kwargs.update(dict(gpus=env_args.multi_gpu, strategy="dp"))
    else:
        trainer_kwargs.update(dict(gpus=[env_args.device_num]) if env_args.is_cuda else dict())

    if env_args.debug:
        trainer_kwargs.update({'log_every_n_steps': 1})
        trainer_kwargs.update({f'limit_{t}_batches': 3 for t in ['train', 'val']})

    if env_args.enable_checkpointing:
        callbacks.append(ModelCheckpoint(monitor='validate_accuracy', mode='max'))

    if additional_callbacks is not None:
        callbacks.extend(additional_callbacks)

    return pl.Trainer(callbacks=callbacks, **trainer_kwargs)


class LitVGG(pl.LightningModule):
    def __init__(self, arch_args: ArchitectureArgs, opt_args: OptimizationArgs, data_args: DataArgs):
        """A basic CNN, based on the VGG architecture (and some variants).

        Args:
            arch_args: The arguments for the architecture.
            opt_args: The arguments for the optimization process.
            data_args: The arguments for the input data.
        """
        super(LitVGG, self).__init__()
        blocks, n_features = get_vgg_blocks(configs[arch_args.model_name], data_args.n_channels, data_args.spatial_size,
                                            arch_args.kernel_size, arch_args.padding, arch_args.use_batch_norm,
                                            arch_args.bottle_neck_dimension, arch_args.pool_as_separate_blocks,
                                            arch_args.shuffle_blocks_output, arch_args.spatial_shuffle_only,
                                            arch_args.fixed_permutation_per_block)
        self.features = nn.Sequential(*blocks)
        self.mlp = get_mlp(input_dim=n_features, output_dim=data_args.n_classes,
                           n_hidden_layers=arch_args.mlp_n_hidden_layers,
                           hidden_dimensions=arch_args.mlp_hidden_dim, use_batch_norm=arch_args.use_batch_norm,
                           shuffle_blocks_output=arch_args.shuffle_blocks_output,
                           fixed_permutation_per_block=arch_args.fixed_permutation_per_block)
        self.loss = torch.nn.CrossEntropyLoss()

        self.arch_args: ArchitectureArgs = arch_args
        self.opt_args: OptimizationArgs = opt_args
        self.data_args: DataArgs = data_args

        self.save_hyperparameters(arch_args.dict())
        self.save_hyperparameters(opt_args.dict())
        self.save_hyperparameters(data_args.dict())

        self.num_blocks = len(self.features) + len(self.mlp)

        self.kernel_sizes: List[int] = self.init_kernel_sizes()
        self.shapes: List[tuple] = self.init_shapes()

    def forward(self, x: torch.Tensor):
        """Performs a forward pass.

        Args:
            x: The input tensor.

        Returns:
            The output of the model, which is logits for the different classes.
        """
        features = self.features(x)
        logits = self.mlp(features)
        return logits

    def should_regularize(self):
        """
        Returns:
            True if we should regularize, False if not (depends on the argument `lasso_regularizer_coefficient`).
        """
        return (
                isinstance(self.arch_args.lasso_regularizer_coefficient, list)
                or
                (self.arch_args.lasso_regularizer_coefficient > 0)
        )

    def get_regularization_loss(self):
        """
        Returns:
            The regularization loss, which is the sum of the l1 norm of the weights of linear/conv layers
            in the model, each multiplied by the corresponding regularization factor.
        """
        blocks = list(self.features) + list(self.mlp)
        num_blocks_with_weights = len([block for block in blocks if len(list(block.parameters())) > 0])
        regularization_coefficients = copy.deepcopy(self.arch_args.lasso_regularizer_coefficient)
        if not isinstance(regularization_coefficients, list):
            regularization_coefficients = [regularization_coefficients] * num_blocks_with_weights
        assert len(regularization_coefficients) == num_blocks_with_weights

        regularization_losses = list()
        for block in blocks:
            # Take the parameters of any conv/linear layer in the block,
            # which essentially excludes the parameters of the batch-norm layer that shouldn't be regularized.
            parameters = [layer.parameters() for layer in block
                          if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)]
            parameters = list(itertools.chain.from_iterable(parameters))
            if len(parameters) == 0:
                continue

            weights_l1_norm = sum(w.abs().sum() for w in parameters)
            regularization_losses.append(regularization_coefficients.pop(0) * weights_l1_norm)

        assert len(regularization_coefficients) == 0

        return sum(regularization_losses)

    def shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: RunningStage):
        """Performs train/validation step, depending on the given `stage`.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            stage: Indicating if this is a training-step or a validation-step.

        Returns:
            The loss.
        """
        inputs, labels = batch
        logits = self(inputs)
        loss = self.loss(logits, labels)
        predictions = torch.argmax(logits, dim=1)
        accuracy = torch.sum(labels == predictions).item() / len(labels)

        self.log(f'{stage.value}_loss', loss)
        self.log(f'{stage.value}_accuracy', accuracy, on_epoch=True, on_step=False)

        if self.should_regularize():
            regularization_loss = self.get_regularization_loss()
            loss += regularization_loss
            self.log(f'{stage.value}_reg_loss', regularization_loss)

        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Performs a training step.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            batch_idx: The index of the batch in the dataset.

        Returns:
            The loss.
        """
        return self.shared_step(batch, RunningStage.TRAINING)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Performs a validation step.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            batch_idx: The index of the batch in the dataset.
        """
        self.shared_step(batch, RunningStage.VALIDATING)

    def get_sub_model(self, i: int) -> nn.Sequential:
        """Extracts a sub-model up to the given layer index.

        Args:
            i: The maximal index to take in the sub-model

        Returns:
            The sub-model.
        """
        if i < len(self.features):
            sub_model = self.features[:i]
        else:
            j = len(self.features) - i  # This is the index in the mlp
            sub_model = nn.Sequential(*(list(self.features) + list(self.mlp[:j])))

        return sub_model

    def configure_optimizers(self):
        """Configure the optimizer and the learning-rate scheduler for the training process.

        Returns:
            A dictionary containing the optimizer and learning-rate scheduler.
        """
        optimizer = torch.optim.SGD(self.parameters(),
                                    self.opt_args.learning_rate,
                                    self.opt_args.momentum,
                                    weight_decay=self.opt_args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.opt_args.learning_rate_decay_steps,
                                                         gamma=self.opt_args.learning_rate_decay_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def init_kernel_sizes(self) -> List[int]:
        """Initialize the kernel-size of each block in the model.

        Returns:
            A list of integers with the same length as `self.features`,
            where the i-th element is the kernel size of the i-th block.
        """
        kernel_sizes = list()
        for i in range(len(self.features)):
            kernel_size = get_vgg_model_kernel_size(self, i)
            if isinstance(kernel_size, tuple):
                assert kernel_size[0] == kernel_size[1], "Only square patches are supported"
                kernel_size = kernel_size[0]
            kernel_sizes.append(kernel_size)
        return kernel_sizes

    @torch.no_grad()
    def init_shapes(self) -> List[Tuple[int, int, int]]:
        """Initialize the input shapes of each block in the model.

        Returns:
            A list of shapes, with the same length as the sum of `self.features` and `self.mlp`.
        """
        shapes = list()
        batch_size = 4  # Need batch_size greater than 1 to propagate through BatchNorm layers

        x = torch.rand(batch_size, self.data_args.n_channels, self.data_args.spatial_size, self.data_args.spatial_size)
        x = x.to(self.device)

        for block in self.features:
            shapes.append(tuple(x.shape[1:]))
            x = block(x)

        x = x.flatten(start_dim=1)

        for block in self.mlp:
            shapes.append(tuple(x.shape[1:]))
            x = block(x)

        shapes.append(tuple(x.shape[1:]))  # This is the output shape

        return shapes


class LitCNN(pl.LightningModule):

    def get_model_config(self, model_name: str, alpha: int, spatial_size: int):
        """Gets the model configuration, determining the architecture.

        This is taken from the paper
        "Towards Learning Convolutions from Scratch", Behnam Neyshabur, Google.
        and the exact architectures are defined in Appendix A.1
        https://proceedings.neurips.cc/paper/2020/file/5c528e25e1fdeaf9d8160dc24dbf4d60-Supplemental.pdf

        Args:
            model_name: The model name, should be {D,S}-{CONV,FC}.
                D stands for "deep", S for "shallow", CONV stands for
                convolutional-network and FC for fully-connected.
            alpha: Denotes the number of base channels which also determines
                the total number of parameters of the architecture.
            spatial_size: The spatial size of the input images (e.g. 32 for CIFAR-10/100).

        Returns: A 4-tuple containing
            conv_channels: A list containing the number of channels for each convolution layer.
                Empty list means that there are no convolution layers at all (i.e. model is fully-connected).
            kernel_sizes: A single integer determining the kernel-size of each convolution-layer.
                 Also here an empty list is returned for fully-connected architecture.
            strides: A list of integers of the same length as `conv_channels`
                determining the stride in each convolutional layer.
                Also here an empty list is returned for fully-connected architecture.
            linear_channels: A list of integers determining the number of channels in each linear layer in the MLP.
        """
        a = alpha
        s = spatial_size

        if model_name == 'D-CONV':
            conv_channels = [a,
                             2 * a,
                             2 * a,
                             4 * a,
                             4 * a,
                             8 * a,
                             8 * a,
                             16 * a]
            kernel_sizes = [3] * 8
            strides = [1, 2] * 4
            linear_channels = [64 * a]
        elif model_name == 'S-CONV':
            conv_channels = [a]
            kernel_sizes = [9]
            strides = [2]
            linear_channels = [24 * a]
        elif model_name == 'D-CONV++':  # Like D-CONV but has another 2 layers which fits ImageNet better
            conv_channels = [a,  # because the final spatial resolution is 7x7 instead of 14x14
                             2 * a,
                             2 * a,
                             4 * a,
                             4 * a,
                             8 * a,
                             8 * a,
                             16 * a,
                             16 * a,
                             32 * a]
            kernel_sizes = [3] * 10
            strides = [1, 2] * 5
            linear_channels = [64 * a]
        elif model_name == 'D-CONV-ResNet18-style':
            conv_channels = [a,
                             a, a, a, a,  # a should be 64
                             2 * a, 2 * a, 2 * a, 2 * a,  # 2*a should be 128
                             4 * a, 4 * a, 4 * a, 4 * a,  # 4*a should be 256
                             8 * a, 8 * a, 8 * a, 8 * a  # 8*a should be 512
                             ]
            kernel_sizes = [7] + [3] * 16
            strides = [2,  # output-size 112x112
                       2, 1, 1, 1,  # output-size 56x56
                       2, 1, 1, 1,  # output-size 28x28
                       2, 1, 1, 1,  # output-size 14x14
                       2, 1, 1, 1  # output-size 7x7
                       ]
            linear_channels = []
        elif model_name == 'D-CONV-ResNet18-style-1st-block-stride-1':
            conv_channels = [a,
                             a, a, a, a,  # a should be 64
                             2 * a, 2 * a, 2 * a, 2 * a,  # 2*a should be 128
                             4 * a, 4 * a, 4 * a, 4 * a,  # 4*a should be 256
                             8 * a, 8 * a, 8 * a, 8 * a  # 8*a should be 512
                             ]
            kernel_sizes = [7] + [3] * 16
            strides = [2,  # output-size 112x112
                       1, 1, 1, 1,  # output-size 56x56, because we use max-pool beforehand
                       2, 1, 1, 1,  # output-size 28x28
                       2, 1, 1, 1,  # output-size 14x14
                       2, 1, 1, 1  # output-size 7x7
                       ]
            linear_channels = []
        elif model_name == 'D-FC':
            conv_channels, kernel_sizes, strides = [], [], []
            linear_channels = [s ** 2 * a,
                               int(s ** 2 * a / 2),
                               int(s ** 2 * a / 2),
                               int(s ** 2 * a / 4),
                               int(s ** 2 * a / 4),
                               int(s ** 2 * a / 8),
                               int(s ** 2 * a / 8),
                               int(s ** 2 * a / 16),
                               64 * a]
        elif model_name == 'S-FC':
            conv_channels, kernel_sizes, strides = [], [], []
            linear_channels = [int(s ** 2 * a / 4),
                               24 * a]
        else:
            raise NotImplementedError(f'Model {model_name} is not recognized.')

        return conv_channels, linear_channels, kernel_sizes, strides

    def __init__(self, arch_args: ArchitectureArgs, opt_args: OptimizationArgs, data_args: DataArgs):
        """A basic CNN, containing only convolutional / linear layers (+ BatchNorm and ReLU).

        The idea is taken from "Towards Learning Convolutions from Scratch", Behnam Neyshabur, Google.

        > One challenge in studying the inductive bias of convolutions is that the existence of other components
        > such as pooling and residual connections makes it difficult to isolate the effect of convolutions
        > in modern architectures.
        > ...
        > To this end, below we propose two all-convolutional networks to overcome the discussed issues.

        Args:
            arch_args: The arguments for the architecture.
            opt_args: The arguments for the optimization process.
            data_args: The arguments for the input data.
        """
        super(LitCNN, self).__init__()
        self.blocks = get_cnn(*self.get_model_config(arch_args.model_name, arch_args.alpha, data_args.spatial_size),
                              paddings=arch_args.padding,
                              shuffle_outputs=arch_args.shuffle_blocks_output,
                              spatial_only=arch_args.spatial_shuffle_only,
                              fixed_permutation=arch_args.fixed_permutation_per_block,
                              replace_with_linear=arch_args.replace_with_linear,
                              replace_with_bottleneck=arch_args.replace_with_bottleneck,
                              randomly_sparse_connected_fractions=arch_args.randomly_sparse_connected_fractions,
                              adaptive_avg_pool_before_mlp=arch_args.adaptive_avg_pool_before_mlp,
                              max_pool_after_first_conv=arch_args.max_pool_after_first_conv,
                              in_spatial_size=data_args.spatial_size,
                              in_channels=data_args.n_channels,
                              n_classes=data_args.n_classes)
        self.loss = torch.nn.CrossEntropyLoss()

        self.arch_args = arch_args
        self.opt_args = opt_args
        self.save_hyperparameters(arch_args.dict())
        self.save_hyperparameters(opt_args.dict())
        self.save_hyperparameters(data_args.dict())

    def forward(self, x: torch.Tensor):
        """Performs a forward pass.

        Args:
            x: The input tensor.

        Returns:
            The output of the model, which is logits for the different classes.
        """
        return self.blocks(x)

    @staticmethod
    def is_list_or_positive_number(arg: Union[list, float]):
        return isinstance(arg, list) or (arg > 0)

    def should_regularize_l2(self):
        """
        Returns:
            True if we should regularize, False if not (depends on the argument `lasso_regularizer_coefficient`).
        """
        return LitCNN.is_list_or_positive_number(self.arch_args.l2_regularizer_coefficient)

    def should_regularize_lasso(self):
        """
        Returns:
            True if we should lasso regularize,
            False if not (depends on the argument `lasso_regularizer_coefficient`).
        """
        return LitCNN.is_list_or_positive_number(self.arch_args.lasso_regularizer_coefficient)

    def should_zero_out_low_weights(self):
        """
        Returns:
            True iff we should zero-out low weights, like done in the "beta-lasso" algorithm in
            "Towards Learning Convolutions from Scratch" (depends on the argument `beta_lasso_coefficient`).
        """
        return LitCNN.is_list_or_positive_number(self.arch_args.beta_lasso_coefficient)

    @staticmethod
    def get_parameters_of_conv_or_linear_layer_in_block(block):
        # Take the parameters of any conv/linear layer in the block,
        # which essentially excludes the parameters of the batch-norm layer that shouldn't be regularized.
        parameters_generators = [layer.parameters() for layer in block if
                                 isinstance(layer, nn.Conv2d) or
                                 isinstance(layer, nn.Linear) or
                                 isinstance(layer, RandomlySparseConnected)]
        parameters = list(itertools.chain.from_iterable(parameters_generators))
        assert len(parameters) > 0, f'Every block should contain a convolutional / linear layer.\nblock is\n{block}'
        return parameters

    def get_regularization_loss(self):
        """
        Returns:
            The regularization loss, which is the sum of the l1 norm of the weights of linear/conv layers
            in the model, each multiplied by the corresponding regularization factor.
        """
        lasso_regularization_coefficients = get_list_of_arguments(self.arch_args.lasso_regularizer_coefficient,
                                                                  len(self.blocks))
        l2_regularization_coefficients = get_list_of_arguments(self.arch_args.l2_regularizer_coefficient,
                                                               len(self.blocks))

        regularization_losses = list()
        for block, lasso, l2 in zip(self.blocks, lasso_regularization_coefficients, l2_regularization_coefficients):
            parameters = LitCNN.get_parameters_of_conv_or_linear_layer_in_block(block)  # TODO consider excluding bias
            if lasso > 0:
                weights_l1_norm = sum(w.abs().sum() for w in parameters)
                regularization_losses.append(lasso * weights_l1_norm)
            if l2 > 0:
                weights_l2_norm = sum(torch.square(w).sum() for w in parameters)
                regularization_losses.append(l2 * weights_l2_norm)

        return sum(regularization_losses)

    def shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: RunningStage):
        """Performs train/validation step, depending on the given `stage`.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            stage: Indicating if this is a training-step or a validation-step.

        Returns:
            The loss.
        """
        inputs, labels = batch
        logits = self(inputs)
        loss = self.loss(logits, labels)
        predictions = torch.argmax(logits, dim=1)
        accuracy = torch.sum(labels == predictions).item() / len(labels)

        self.log(f'{stage.value}_loss', loss)
        self.log(f'{stage.value}_accuracy', accuracy, on_epoch=True, on_step=False)

        if self.should_regularize_lasso() or self.should_regularize_l2():
            regularization_loss = self.get_regularization_loss()
            loss += regularization_loss
            self.log(f'{stage.value}_reg_loss', regularization_loss)

        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Performs a training step.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            batch_idx: The index of the batch in the dataset.

        Returns:
            The loss.
        """
        return self.shared_step(batch, RunningStage.TRAINING)

    def on_train_batch_end(self, outputs, batch, batch_idx: int, unused: Optional[int] = 0) -> None:
        if self.should_zero_out_low_weights():
            self.zero_out_low_weights()
        self.log_nonzero_parameters_count()

    def zero_out_low_weights(self):
        # The notations "beta" and "lambda" are taken from "Towards Learning Convolutions from Scratch".
        betas = get_list_of_arguments(self.arch_args.beta_lasso_coefficient, len(self.blocks))
        lambdas = get_list_of_arguments(self.arch_args.lasso_regularizer_coefficient, len(self.blocks))

        for block, beta, lmbda in zip(self.blocks, betas, lambdas):
            threshold = beta * lmbda  # lambda written with typo in purpose (since it's python's reserved word)
            if threshold == 0:  # In other words - if beta or lambda is zero, move on.
                continue
            for parameter in LitCNN.get_parameters_of_conv_or_linear_layer_in_block(block):
                with torch.no_grad():
                    # To understand why this is done in the context-manager `no_grad` see here:
                    # https://medium.com/@mrityu.jha/understanding-the-grad-of-autograd-fc8d266fd6cf
                    parameter[parameter.abs() < threshold] = 0

    def log_nonzero_parameters_count(self):
        for i, block in enumerate(self.blocks):
            for j, parameter in enumerate(LitCNN.get_parameters_of_conv_or_linear_layer_in_block(block)):
                nonzero_fraction = torch.count_nonzero(parameter).item() / parameter.numel()
                self.log(f'block-{i}-param-{j}-nonzero-fraction', nonzero_fraction, on_epoch=True, on_step=False)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Performs a validation step.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            batch_idx: The index of the batch in the dataset.
        """
        self.shared_step(batch, RunningStage.VALIDATING)

    def configure_optimizers(self):
        """Configure the optimizer and the learning-rate scheduler for the training process.

        Returns:
            A dictionary containing the optimizer and learning-rate scheduler.
        """
        optimizer = torch.optim.SGD(self.parameters(),
                                    self.opt_args.learning_rate,
                                    self.opt_args.momentum,
                                    weight_decay=self.opt_args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.opt_args.learning_rate_decay_steps,
                                                         gamma=self.opt_args.learning_rate_decay_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


class LitMLP(pl.LightningModule):
    def __init__(self, arch_args: ArchitectureArgs, opt_args: OptimizationArgs, data_args: DataArgs):
        """A basic MLP, which consists of multiple linear layers with ReLU in-between.

        Args:
            arch_args: The arguments for the architecture.
            opt_args: The arguments for the optimization process.
            data_args: The arguments for the input data.
        """
        super(LitMLP, self).__init__()
        self.input_dim = data_args.n_channels * data_args.spatial_size ** 2
        self.output_dim = data_args.n_classes
        self.n_hidden_layers = arch_args.mlp_n_hidden_layers
        self.hidden_dim = arch_args.mlp_hidden_dim
        self.mlp = get_mlp(self.input_dim, self.output_dim, self.n_hidden_layers, self.hidden_dim,
                           use_batch_norm=arch_args.use_batch_norm,
                           shuffle_blocks_output=arch_args.shuffle_blocks_output,
                           fixed_permutation_per_block=arch_args.fixed_permutation_per_block)
        self.loss = torch.nn.CrossEntropyLoss()

        self.arch_args = arch_args
        self.opt_args = opt_args
        self.save_hyperparameters(arch_args.dict())
        self.save_hyperparameters(opt_args.dict())
        self.save_hyperparameters(data_args.dict())

        self.num_blocks = len(self.mlp)

    def forward(self, x: torch.Tensor):
        return self.mlp(x)

    def should_regularize(self):
        """
        Returns:
            True if we should regularize, False if not (depends on the argument `lasso_regularizer_coefficient`).
        """
        return (
                isinstance(self.arch_args.lasso_regularizer_coefficient, list)
                or
                (self.arch_args.lasso_regularizer_coefficient > 0)
        )

    def get_regularization_loss(self):
        """
        Returns:
            The regularization loss, which is the sum of the l1 norm of the weights of linear/conv layers
            in the model, each multiplied by the corresponding regularization factor.
        """
        blocks = list(self.mlp)
        num_blocks_with_weights = len([block for block in blocks if len(list(block.parameters())) > 0])
        regularization_coefficients = copy.deepcopy(self.arch_args.lasso_regularizer_coefficient)
        if not isinstance(regularization_coefficients, list):
            regularization_coefficients = [regularization_coefficients] * num_blocks_with_weights
        assert len(regularization_coefficients) == num_blocks_with_weights

        regularization_losses = list()
        for block in blocks:
            # Take the parameters of any conv/linear layer in the block,
            # which essentially excludes the parameters of the batch-norm layer that shouldn't be regularized.
            parameters = [layer.parameters() for layer in block if isinstance(layer, nn.Linear)]
            parameters = list(itertools.chain.from_iterable(parameters))
            if len(parameters) == 0:
                continue

            # flattened_parameters = [parameter.view(-1) for parameter in parameters]
            weights_l1_norm = sum(w.abs().sum() for w in parameters)
            regularization_losses.append(regularization_coefficients.pop(0) * weights_l1_norm)

        assert len(regularization_coefficients) == 0

        return sum(regularization_losses)

    def shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: RunningStage):
        """Performs train/validation step, depending on the given `stage`.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            stage: Indicating if this is a training-step or a validation-step.

        Returns:
            The loss.
        """
        inputs, labels = batch
        logits = self(inputs)
        loss = self.loss(logits, labels)
        predictions = torch.argmax(logits, dim=1)
        accuracy = torch.sum(labels == predictions).item() / len(labels)

        self.log(f'{stage.value}_loss', loss)
        self.log(f'{stage.value}_accuracy', accuracy, on_epoch=True, on_step=False)

        if self.should_regularize():
            regularization_loss = self.get_regularization_loss()
            loss += regularization_loss
            self.log(f'{stage.value}_reg_loss', regularization_loss)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, RunningStage.TRAINING)
        return loss

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, RunningStage.VALIDATING)

    def get_sub_model(self, i: int) -> nn.Sequential:
        return self.mlp[:i]

    def configure_optimizers(self):
        """Configure the optimizer and the learning-rate scheduler for the training process.

        Returns:
            A dictionary containing the optimizer and learning-rate scheduler.
        """
        optimizer = torch.optim.SGD(self.parameters(),
                                    self.opt_args.learning_rate,
                                    self.opt_args.momentum,
                                    weight_decay=self.opt_args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.opt_args.learning_rate_decay_steps,
                                                         gamma=self.opt_args.learning_rate_decay_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


def unwatch_model(model: nn.Module):
    """Unwatch a model, to be watched in the next training-iteration.

    Prevents wandb error which they have a TO-DO for fixing in wandb/sdk/wandb_watch.py:123
    "  TO-DO: we should also remove recursively model._wandb_watch_called  "
    ValueError: You can only call `wandb.watch` once per model.
    Pass a new instance of the model if you need to call wandb.watch again in your code.

    Args:
        model: The model to unwatch.
    """
    wandb.unwatch(model)
    for module in model.modules():
        if hasattr(module, "_wandb_watch_called"):
            delattr(module, "_wandb_watch_called")
    if hasattr(model, "_wandb_watch_called"):
        delattr(model, "_wandb_watch_called")
    wandb.finish()
