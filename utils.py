import argparse
import math
import copy
import datetime
import os
import sys
import time
import itertools
import yaml
import torch
import torchvision
import wandb

import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from functools import partial
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from typing import List, Optional, Dict, Callable, Tuple, Union
from loguru import logger
from datetime import timedelta
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize

from consts import CLASSES, LOGGER_FORMAT, DATETIME_STRING_FORMAT


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
    """Gets arguments pydantic class according to the argparse object (possibly including the yaml config).
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


def get_mlp(input_dim: int, output_dim: int, n_hidden_layers: int = 0,
            hidden_dimensions: Union[int, List[int]] = 0,
            use_batch_norm: bool = False, organize_as_blocks: bool = True,
            shuffle_blocks_output: Union[bool, List[bool]] = False,
            fixed_permutation_per_block: bool = False) -> torch.nn.Sequential:
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

    Returns:
        A sequential model which is the constructed MLP.
    """
    layers: List[torch.nn.Module] = list()
    if isinstance(shuffle_blocks_output, list):
        shuffle_blocks_output = any(shuffle_blocks_output)
    if not isinstance(hidden_dimensions, list):
        hidden_dimensions = [hidden_dimensions] * n_hidden_layers
    assert len(hidden_dimensions) == n_hidden_layers

    in_features = input_dim
    for i, hidden_dim in enumerate(hidden_dimensions):
        block_layers: List[nn.Module] = list()
        out_features = hidden_dim

        # Begins with a Flatten layer. It's useful when the input is 4D from a conv layer, and harmless otherwise.
        if i == 0:
            block_layers.append(nn.Flatten())

        block_layers.append(torch.nn.Linear(in_features, out_features))
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

    final_layer = nn.Linear(in_features, output_dim)
    if organize_as_blocks:
        final_layer = nn.Sequential(final_layer)

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
            in_spatial_size: int = 32,
            in_channels: int = 3) -> torch.nn.Sequential:
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
    :param replace_with_linear: Whether to replace each conv layer with a linear layer of the same expressivity.
    :param in_spatial_size: Will be used to infer input dimension for the first affine layer.
    :param in_channels: Number of channels in the input tensor.
    :return: A sequential model which is the constructed CNN.
    """
    blocks: List[nn.Sequential] = list()

    use_max_pool = get_list_of_arguments(use_max_pool, len(conv_channels), default=False)
    shuffle_outputs = get_list_of_arguments(shuffle_outputs, len(conv_channels), default=False)
    strides = get_list_of_arguments(strides, len(conv_channels), default=1)
    kernel_sizes = get_list_of_arguments(kernel_sizes, len(conv_channels), default=3)
    paddings = get_list_of_arguments(paddings, len(conv_channels),
                                     default=[kernel_size // 2 for kernel_size in kernel_sizes])
    spatial_only_list = get_list_of_arguments(spatial_only, len(conv_channels), default=True)
    fixed_permutation_list = get_list_of_arguments(fixed_permutation, len(conv_channels), default=True)
    replace_with_linear = get_list_of_arguments(replace_with_linear, len(conv_channels), default=False)

    zipped_args = zip(conv_channels, paddings, strides, kernel_sizes, use_max_pool,
                      shuffle_outputs, spatial_only_list, fixed_permutation_list, replace_with_linear)
    for out_channels, padding, stride, kernel_size, pool, shuffle, spatial, fixed, linear in zipped_args:
        block_layers: List[nn.Module] = list()

        out_spatial_size = int(math.floor((in_spatial_size + 2 * padding - kernel_size) / stride + 1))
        if pool:
            out_spatial_size = int(math.floor(out_spatial_size / 2))

        if linear:
            block_layers.append(nn.Flatten())
            block_layers.append(nn.Linear(in_features=in_channels * in_spatial_size ** 2,
                                          out_features=out_channels * out_spatial_size**2))
            block_layers.append(View(shape=(out_channels, out_spatial_size, out_spatial_size)))
        else:
            block_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))

        block_layers.append(nn.BatchNorm2d(out_channels))
        block_layers.append(nn.ReLU())

        if pool:
            block_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if shuffle:
            block_layers.append(ShuffleTensor(out_spatial_size, out_channels, spatial, fixed))

        blocks.append(nn.Sequential(*block_layers))
        in_channels = out_channels
        in_spatial_size = out_spatial_size

    mlp = get_mlp(input_dim=in_channels * (in_spatial_size ** 2),
                  output_dim=len(CLASSES),
                  n_hidden_layers=len(linear_channels),
                  hidden_dimensions=linear_channels,
                  use_batch_norm=True,
                  organize_as_blocks=True)
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
    n_images = dataloader.dataset.data.shape[0]
    patch_shape = dataloader.dataset.data.shape[1:]

    if len(patch_shape) == 2:  # Add dimension of channels which will be 1
        patch_shape += (1,)

    patch_shape = np.roll(patch_shape, shift=1)  # In the dataset it's H x W x C but in the model it's C x H x W
    if existing_model is not None:
        device = get_model_device(existing_model)
        patch_shape = get_model_output_shape(existing_model, dataloader)

    if len(patch_shape) > 1:
        assert len(patch_shape) == 3 and (patch_shape[1] == patch_shape[2]), "Should be C x H x W where H = W"
        spatial_size = patch_shape[-1]
        if patch_size == -1:  # -1 means the patch size is the whole size of the image
            patch_size = spatial_size
        n_patches_per_row_or_col = spatial_size - patch_size + 1
        patch_shape = (patch_shape[0],) + 2 * (patch_size,)
    else:
        assert patch_size == -1, "When working with fully-connected the patch 'size' must be -1 i.e. the whole size."
        n_patches_per_row_or_col = 1

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


class Accumulator:
    """
    Accumulate loss and correct predictions of an interval,
    to calculate later mean-loss & accuracy.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_sum: float = 0
        self.corrects_sum: int = 0
        self.total_samples: int = 0
        self.begin_time: float = time.time()

        # These are for training with SSL & DGL combined, to examine the two different objectives separately.
        self.pred_loss_sum: float = 0
        self.ssl_loss_sum: float = 0

    def update(self, mean_loss: float, num_corrects: int, n_samples: int,
               mean_pred_loss: float = 0, mean_ssl_loss: float = 0):
        self.loss_sum += mean_loss * n_samples
        self.pred_loss_sum += mean_pred_loss * n_samples
        self.ssl_loss_sum += mean_ssl_loss * n_samples
        self.corrects_sum += num_corrects
        self.total_samples += n_samples

    def get_mean_loss(self) -> float:
        return self.loss_sum / self.total_samples

    def get_mean_pred_loss(self) -> float:
        return self.pred_loss_sum / self.total_samples

    def get_mean_ssl_loss(self) -> float:
        return self.ssl_loss_sum / self.total_samples

    def get_accuracy(self) -> float:
        return 100 * (self.corrects_sum / self.total_samples)

    def get_time(self) -> float:
        return time.time() - self.begin_time

    def get_dict(self, prefix='') -> Dict[str, float]:
        d = {f'{prefix}_accuracy': self.get_accuracy(),
             f'{prefix}_loss': self.get_mean_loss()}

        if self.get_mean_pred_loss() > 0:
            d[f'{prefix}_pred_loss'] = self.get_mean_pred_loss()
        if self.get_mean_ssl_loss() > 0:
            d[f'{prefix}_ssl_loss'] = self.get_mean_ssl_loss()

        return d


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
                           modules_accumulators: List[Optional[Accumulator]],
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
    Calcuates batch-wise dot-product, used
    https://github.com/pytorch/pytorch/issues/18027#issuecomment-473119948
    as a reference.
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


def perform_train_step_regular(model, inputs, labels, criterion, optimizer):
    """
    Perform a regular train-step:
    (1) Feed the inputs (i.e. minibatch images) to the model.
    (2) Get the predictions (i.e. classes' scores).
    (3) Calculate the loss with respect to the labels.
    (4) Perform backward pass and a single optimizer step.

    :param model: The model.
    :param inputs: The inputs.
    :param labels: The labels.
    :param criterion: The criterion.
    :param optimizer: The optimizer.
    :return: The loss of this train-step, as well as the predictions.
    """
    optimizer.zero_grad()

    outputs = model(inputs)
    _, predictions = torch.max(outputs, dim=1)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()

    return loss.item(), predictions


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


def get_optim(model, optimizer_params, is_dgl, use_last_gradient=False):
    """
    Return the optimizer (or plural optimizers) to train the given model.
    If the model is trained with DGL there are several optimizers,
    one for each block in the model (which is optimizing the block
    parameters as well as the corresponding auxiliary network parameters).

    :param model: The model.
    :param optimizer_params: A dictionary describing the optimizer parameters: learning-rate, momentum and weight-decay.
    :param is_dgl: Whether this model is trained with DGL or not.
    :return: An optimizer, or a list of optimizers (if the model is trained with DGL).
    """
    optimizer_type: str = optimizer_params.pop('optimizer_type')
    if optimizer_type == 'Adam':
        optimizer_constuctor = torch.optim.Adam
        optimizer_params.pop('momentum')  # momentum is relevant only for SGD, not for Adam.
    elif optimizer_type == 'SGD':
        optimizer_constuctor = torch.optim.SGD
    else:
        raise ValueError(f'optimizer_type {optimizer_type} should be \'Adam\' or \'SGD\'.')

    if use_last_gradient:
        return optimizer_constuctor(model.parameters(), **optimizer_params)
    if is_dgl:
        optimizers = list()

        for i in range(len(model.blocks)):
            if len(list(model.blocks[i].parameters())) == 0:
                optimizers.append(None)
            else:
                parameters_to_train = itertools.chain(
                    model.blocks[i].parameters(),
                    model.auxiliary_nets[i].parameters(),
                    model.ssl_auxiliary_nets[i].parameters() if model.ssl_auxiliary_nets is not None else list()
                )

                optimizer = optimizer_constuctor(parameters_to_train, **optimizer_params)
                optimizers.append(optimizer)

        return optimizers
    else:
        return optimizer_constuctor(model.parameters(), **optimizer_params)


def train_local_model(model, dataloaders, criterion, optimizer, device,
                      num_epochs=25, log_interval=100,
                      is_dgl=False, is_ssl=False,
                      ssl_criterion=None, pred_loss_weight=1, ssl_loss_weight=0.1,
                      first_trainable_block=0, shift_ssl_labels=False,
                      is_direct_global=False, last_gradient_weight: float = 0.5,
                      use_last_gradient=False):
    """
    A general function to train a model and return the best model found.

    :param model: the model to train
    :param criterion: which loss to train on
    :param optimizer: the optimizer to train with
    :param dataloaders: the dataloaders to feed the model
    :param device: which device to train on
    :param num_epochs: how many epochs to train
    :param log_interval: How many training/testing steps between each logging (to wandb).
    :param is_dgl: Whether this model is trained using DGL (affects the optimizer and the train-step functionality).
    :param is_ssl: Whether this model is trained using self-supervised local loss (predicting the shifted image).
    :param ssl_criterion:
    :param pred_loss_weight: When combining with DGL, the weight for the scores' prediction loss.
    :param ssl_loss_weight: When combining with DGL, the weight for the SSL prediction loss.
    :param first_trainable_block: Can be used to freeze a few layers in their initial weights.
    :param shift_ssl_labels: Whether to shift the SSL labels or keep them the original images.
    :param is_direct_global: Whether to use direct global gradient or not.
    :param use_last_gradient: Whether to use last gradient or not.
                              This should be the same in theory as `is_direct_global`
                              but it's implemented quite differently.
    :param last_gradient_weight: Weight of the last gradient in each intermediate gradient calculator.
    :return: the model with the lowest test error
    """
    best_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    training_step = 0
    total_time = 0
    interval_accumulator = Accumulator()
    epoch_accumulator = Accumulator()
    modules_accumulators = None if (not is_dgl) else [Accumulator() if (aux_net is not None) else None
                                                      for aux_net in model.auxiliary_nets]

    for epoch in range(num_epochs):
        model.train()
        epoch_accumulator.reset()

        for inputs, labels in dataloaders['train']:
            training_step += 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            train_step_result = perform_train_step(model, inputs, labels, criterion, optimizer, training_step,
                                                   is_dgl, is_ssl, ssl_criterion, pred_loss_weight, ssl_loss_weight,
                                                   first_trainable_block, shift_ssl_labels, is_direct_global,
                                                   modules_accumulators, last_gradient_weight, use_last_gradient)

            accumulator_kwargs = dict()
            if is_ssl:
                loss, predictions, predictions_loss, ssl_loss = train_step_result
                accumulator_kwargs['mean_pred_loss'] = predictions_loss.item() if predictions_loss is not None else 0
                accumulator_kwargs['mean_ssl_loss'] = ssl_loss.item() if ssl_loss is not None else 0
            else:
                loss, predictions = train_step_result

            accumulator_kwargs['mean_loss'] = loss
            accumulator_kwargs['num_corrects'] = torch.sum(torch.eq(predictions, labels.data)).item()
            accumulator_kwargs['n_samples'] = inputs.size(0)  # This equals the batch-size, except in the last minibatch

            epoch_accumulator.update(**accumulator_kwargs)
            interval_accumulator.update(**accumulator_kwargs)

            if training_step % log_interval == 0:
                wandb.log(data=interval_accumulator.get_dict(prefix='train'), step=training_step)
                interval_accumulator.reset()

        if use_last_gradient:
            epoch_test_loss, epoch_test_accuracy = evaluate_model_with_last_gradient(model, criterion,
                                                                                     dataloaders['test'], device,
                                                                                     training_step)
        elif is_dgl:
            epoch_test_loss, epoch_test_accuracy = evaluate_local_model(model, criterion, dataloaders['test'], device,
                                                                        training_step)
        else:
            epoch_test_loss, epoch_test_accuracy = evaluate_model(model, criterion, dataloaders['test'], device)

        wandb.log(data={'test_accuracy': epoch_test_accuracy, 'test_loss': epoch_test_loss}, step=training_step)

        # if the current model reached the best results so far, deep copy the weights of the model.
        if epoch_test_accuracy > best_accuracy:
            best_accuracy = epoch_test_accuracy
            best_weights = copy.deepcopy(model.state_dict())

        epoch_time = epoch_accumulator.get_time()
        total_time += epoch_time
        log_epoch_end(epoch, epoch_time, num_epochs, total_time,
                      epoch_accumulator.get_mean_loss(), epoch_accumulator.get_accuracy(),
                      epoch_test_loss, epoch_test_accuracy)

    logger.info(f'Best test accuracy: {best_accuracy:.2f}%')

    model.load_state_dict(best_weights)  # load best model weights
    return model


def train_model(model: nn.Module,
                dataloaders: Dict[str, torch.utils.data.DataLoader],
                criterion: nn.CrossEntropyLoss,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                epochs: int,
                log_interval: int):
    best_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    training_step = 0
    total_time = 0
    interval_accumulator = Accumulator()
    epoch_accumulator = Accumulator()
    device = get_model_device(model)

    for epoch in range(epochs):
        model.train()
        epoch_accumulator.reset()

        for inputs, labels in dataloaders['train']:
            training_step += 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            loss, predictions = perform_train_step_regular(model, inputs, labels, criterion, optimizer)

            accumulator_kwargs = dict(mean_loss=loss,
                                      num_corrects=torch.sum(torch.eq(predictions, labels.data)).item(),
                                      n_samples=inputs.size(0))  # This equals the batch-size, except in the last batch

            epoch_accumulator.update(**accumulator_kwargs)
            interval_accumulator.update(**accumulator_kwargs)

            if training_step % log_interval == 0:
                wandb.log(data=interval_accumulator.get_dict(prefix='train'), step=training_step)
                # logger.info(f'{training_step=:10d} '
                #             f'loss={interval_accumulator.get_mean_loss():.4f} '
                #             f'acc={interval_accumulator.get_accuracy():.2f}%')
                interval_accumulator.reset()

        epoch_test_loss, epoch_test_accuracy = evaluate_model(model, criterion, dataloaders['test'], device)
        wandb.log(data={'test_accuracy': epoch_test_accuracy, 'test_loss': epoch_test_loss}, step=training_step)

        # if the current model reached the best results so far, deep copy the weights of the model.
        if epoch_test_accuracy > best_accuracy:
            best_accuracy = epoch_test_accuracy
            best_weights = copy.deepcopy(model.state_dict())

        scheduler.step()

        epoch_time = epoch_accumulator.get_time()
        total_time += epoch_time
        log_epoch_end(epoch, epoch_time, epochs, total_time,
                      epoch_accumulator.get_mean_loss(), epoch_accumulator.get_accuracy(),
                      epoch_test_loss, epoch_test_accuracy)

    logger.info(f'Best test accuracy: {best_accuracy:.2f}%')
    model.load_state_dict(best_weights)  # load best model weights


def create_out_dir(parent_out_dir: str) -> str:
    """
    Creates the output directory in the given parent output directory,
    which will be named by the current date and time.
    """
    datetime_string = datetime.datetime.now().strftime(DATETIME_STRING_FORMAT)
    out_dir = os.path.join(parent_out_dir, datetime_string)
    os.mkdir(out_dir)

    return out_dir


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


def log_epoch_end(epoch, epoch_time_elapsed, total_epochs, total_time,
                  epoch_train_loss, epoch_train_accuracy,
                  epoch_test_loss, epoch_test_accuracy):
    total_time += epoch_time_elapsed
    epochs_left = total_epochs - (epoch + 1)
    avg_epoch_time = total_time / (epoch + 1)
    time_left = avg_epoch_time * epochs_left
    total_epochs_n_digits = len(str(total_epochs))
    logger.info(f'Epoch {epoch + 1:0>{total_epochs_n_digits}d}/{total_epochs:0>{total_epochs_n_digits}d} '
                f'({str(timedelta(seconds=epoch_time_elapsed)).split(".")[0]}) | '
                f'ETA {str(timedelta(seconds=time_left)).split(".")[0]} | '
                f'Train '
                f'loss={epoch_train_loss:.4f} '
                f'acc={epoch_train_accuracy:.2f}% | '
                f'Test '
                f'loss={epoch_test_loss:.4f} '
                f'acc={epoch_test_accuracy:.2f}%')


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
