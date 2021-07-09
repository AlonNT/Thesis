"""
This file runs the experiments for patches-based learning, similar to
The Unreasonable Effectiveness of Patches in Deep Convolutional Kernels Methods
(https://arxiv.org/pdf/2101.07528.pdf)

(1) Collect patches from the dataset (to enable calculating later k-nearest-neighbors).
(2) ?Calculates whitening?
(3) Defines the model which will
"""
import argparse
import copy
import os
from functools import partial
from math import ceil

from torchvision.transforms.functional import hflip

import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from loguru import logger
from typing import Callable, Optional
from datetime import timedelta

from consts import CIFAR10_IMAGE_SIZE, N_CLASSES
from utils import (configure_logger,
                   create_out_dir,
                   get_dataloaders,
                   Accumulator,
                   perform_train_step_regular,
                   evaluate_model)


"""
Ideas

Bridge the gap between my implementation and theirs.
* Classifier architecture:
  Theirs:
  - Input image
    3 x 32 x 32
  - Twice (for positive / negative patches):
    + k-nearest-patches-mask
      2048 x 27 x 27
    + AvgPool(kernel_size=5, stride=3, ceil_mode=True)
      2048 x 9 x 9
    + AdaptiveAvgPool(output_size=6)
      2048 x 6 x 6
    + BatchNorm
      2048 x 6 x 6
    + Conv2d(2048, 128, kernel_size=1, stride=1)
      128 x 6 x 6
  - Addition (element-wise) of the two tensors corresponding to positive / negative patches
    128 x 6 x 6
  - Conv2d(128, 10, kernel_size=6, stride=1)
    10 x 1 x 1
  - AdaptiveAvgPool(output_size=1) 
    NO EFFECT since the size is already the target size.
    10 x 1 x 1
 
  Mine:
  - Input image
    3 x 32 x 32
  - Twice (for positive / negative patches):
    + k-nearest-patches-mask
      2048 x 27 x 27
    + AvgPool(kernel_size=5, stride=3, ceil_mode=True)
      2048 x 9 x 9
    + BatchNorm
      2048 x 9 x 9
    + Conv2d(2048, 128, kernel_size=1, stride=1)
      128 x 9 x 9
  - Addition (element-wise) of the two tensors corresponding to positive / negative patches
    128 x 9 x 9
  - Flatten
    10,368
  - Linear(10368, 10)
    10

* In the linear classifier experiments the AvgPool stride is 3 
  and in the 1-hidden-layer (i.e. ReLU in-between) it's 2. 
* Use the same learning-rate and scheduler.
* The data-loader they use for whitening / patches-extraction 
  is images with values in [0,1] (the only transform is ToTensor).
  The data-loader they use for training is unit-gaussian 
  with random cropping and horizontal flipping.

General improvements:
* Make a better use of patches - instead of random patches selection:
  + Cluster them (is it tractable?)
  + Iteratively improve the patches dictionary - random at the beginning 
    and then keep only the most "active" ones.

Experiments:
* Remove the AdaptiveAvgPool which reduce spatial size from 9x9 to 6x6.
* When adding negative batches simply concat and don't duplicate all layers.
* Use ReLU after the bottle-neck layer (conv 1x1) to make it non-linear. They reached accuracy 88.53% at epoch 140/174.
* Check if horizontal flipping of the patches helps.
"""


def calculate_smaller_than_kth_value_mask(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.less(x, x.kthvalue(dim=1, k=k+1, keepdim=True).values)


class ClassifierOnPatchBasedEmbedding(nn.Module):
    def __init__(self,
                 kernel_convolution: torch.Tensor,
                 bias_convolution: torch.Tensor,
                 k_neighbors: int,
                 n_channels: int,
                 add_flipped_patches: bool,
                 add_negative_patches_as_network_branch: bool,
                 add_negative_patches_as_more_patches: bool,
                 use_batch_norm: bool,
                 conv_kernel_size: int,
                 use_avg_pool: bool,
                 pool_size: int,
                 pool_stride: int,
                 use_adaptive_avg_pool: bool,
                 use_relu_after_bottleneck: bool):
        super(ClassifierOnPatchBasedEmbedding, self).__init__()

        self.add_flipped_patches = add_flipped_patches
        self.add_negative_patches_as_network_branch = add_negative_patches_as_network_branch
        self.use_avg_pool = use_avg_pool
        self.use_batch_norm = use_batch_norm
        self.use_adaptive_avg_pool = use_adaptive_avg_pool
        self.use_relu_after_bottleneck = use_relu_after_bottleneck
        self.n_patches = kernel_convolution.shape[0]
        self.embedding_n_channels = (2 ** (int(add_negative_patches_as_more_patches) +
                                           int(add_flipped_patches))) * self.n_patches

        kernel_size = kernel_convolution.shape[-1]
        embedding_spatial_size = CIFAR10_IMAGE_SIZE - kernel_size + 1
        pooled_embedding_dim = ceil((embedding_spatial_size - pool_size) / pool_stride + 1)
        conv_input_spatial_size = pooled_embedding_dim if use_avg_pool else embedding_spatial_size
        if use_adaptive_avg_pool:
            conv_input_spatial_size = 6
        conv_output_spatial_size = conv_input_spatial_size - conv_kernel_size + 1
        intermediate_n_features = n_channels * (conv_output_spatial_size ** 2)                                           

        self.patch_based_embedding = PatchBasedEmbedding(
            kernel_convolution, bias_convolution, k_neighbors, add_flipped_patches,
            add_negative_patches_as_network_branch, add_negative_patches_as_more_patches)
        self.avg_pool = nn.AvgPool2d(pool_size, pool_stride, ceil_mode=True) if use_avg_pool else None
        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=6) if use_adaptive_avg_pool else None
        self.flatten = nn.Flatten()
        self.bottleneck_relu = nn.ReLU() if self.use_relu_after_bottleneck else None
        self.final_layer = nn.Linear(in_features=intermediate_n_features, out_features=N_CLASSES)

        if not self.add_negative_patches_as_network_branch:
            self.batch_norm = nn.BatchNorm2d(self.embedding_n_channels) if use_batch_norm else None
            self.bottle_neck_conv = nn.Conv2d(in_channels=self.embedding_n_channels,
                                              out_channels=n_channels,
                                              kernel_size=conv_kernel_size)
        else:
            self.batch_norm_1 = nn.BatchNorm2d(self.n_patches) if use_batch_norm else None
            self.batch_norm_2 = nn.BatchNorm2d(self.n_patches) if use_batch_norm else None
            self.bottle_neck_conv_1 = nn.Conv2d(in_channels=self.n_patches,
                                                out_channels=n_channels,
                                                kernel_size=conv_kernel_size)
            self.bottle_neck_conv_2 = nn.Conv2d(in_channels=self.n_patches,
                                                out_channels=n_channels,
                                                kernel_size=conv_kernel_size)

    def forward(self, x):
        if not self.add_negative_patches_as_network_branch:
            embedding = self.patch_based_embedding(x)
            if self.use_avg_pool:
                embedding = self.avg_pool(embedding)
            if self.use_adaptive_avg_pool:
                embedding = self.adaptive_avg_pool(embedding)
            if self.use_batch_norm:
                embedding = self.batch_norm(embedding)
            embedding = self.bottle_neck_conv(embedding)
        else:
            embedding1, embedding2 = self.patch_based_embedding(x)

            if self.use_avg_pool:
                embedding1 = self.avg_pool(embedding1)
                embedding2 = self.avg_pool(embedding2)

            if self.use_adaptive_avg_pool:
                embedding1 = self.adaptive_avg_pool(embedding1)
                embedding2 = self.adaptive_avg_pool(embedding2)

            if self.use_batch_norm:
                embedding1 = self.batch_norm_1(embedding1)
                embedding2 = self.batch_norm_2(embedding2)

            embedding1 = self.bottle_neck_conv_1(embedding1)
            embedding2 = self.bottle_neck_conv_2(embedding2)

            # TODO when add_negative_patches_as_network_branch is True, try to perform ReLU on each separate embedding
            # if self.bottleneck_relu:
            #     embedding1 = self.bottleneck_relu(embedding1)
            #     embedding2 = self.bottleneck_relu(embedding2)

            embedding = embedding1 + embedding2

        if self.bottleneck_relu:
            embedding = self.bottleneck_relu(embedding)
        
        embedding_flat = self.flatten(embedding)
        scores = self.final_layer(embedding_flat)

        return scores


class PatchBasedEmbedding(nn.Module):
    """
    Calculating the k-nearest-neighbors is implemented as convolution with bias, as was done in
    The Unreasonable Effectiveness of Patches in Deep Convolutional Kernels Methods
    (https://arxiv.org/pdf/2101.07528.pdf)
    Details can be found in Appendix B (page 13).
    """

    def __init__(self,
                 kernel_convolution: torch.Tensor,
                 bias_convolution: torch.Tensor,
                 k_neighbors: int,
                 add_flipped_patches: bool,
                 add_negative_patches_as_network_branch: bool,
                 add_negative_patches_as_more_patches: bool):
        super(PatchBasedEmbedding, self).__init__()

        self.kernel_convolution = nn.Parameter(kernel_convolution, requires_grad=False)

        # The bias will be added to a tensor of shape (N, n_patches, H, W) to reshaping it to (1, n_patches, 1, 1)
        # will make the addition "broadcastable".
        self.bias_convolution = nn.Parameter(bias_convolution.view(1, -1, 1, 1), requires_grad=False)

        self.k_neighbors = k_neighbors
        self.add_flipped_patches = add_flipped_patches
        self.add_negative_patches_as_network_branch = add_negative_patches_as_network_branch
        self.add_negative_patches_as_more_patches = add_negative_patches_as_more_patches

        if self.add_flipped_patches:
            self.flipped_kernel = nn.Parameter(hflip(self.kernel_convolution), requires_grad=False)
        else:
            self.flipped_kernel = None

    def forward(self, images):
        conv_result_no_bias = F.conv2d(images, self.kernel_convolution)
        squared_distances = -1 * conv_result_no_bias + self.bias_convolution

        calc_negative_patches = self.add_negative_patches_as_network_branch or self.add_negative_patches_as_more_patches
        squared_distances_to_negative_patches = (conv_result_no_bias + self.bias_convolution if calc_negative_patches
                                                 else None)

        all_squared_distances = [squared_distances]

        if self.add_flipped_patches:
            conv_result_to_flipped_kernel_no_bias = F.conv2d(images, self.flipped_kernel)
            squared_distances_to_flipped_patches = -1 * conv_result_to_flipped_kernel_no_bias + self.bias_convolution
            all_squared_distances.append(squared_distances_to_flipped_patches)

        if self.add_negative_patches_as_more_patches:
            all_squared_distances.append(squared_distances_to_negative_patches)
            if self.add_flipped_patches:
                squared_distances_to_flipped_negative_patches = (conv_result_to_flipped_kernel_no_bias +
                                                                 self.bias_convolution)
                all_squared_distances.append(squared_distances_to_flipped_negative_patches)

        if self.add_flipped_patches or self.add_negative_patches_as_more_patches:
            squared_distances_to_all_patches = torch.cat(all_squared_distances, dim=1)
            k_nearest_patches_mask = calculate_smaller_than_kth_value_mask(squared_distances_to_all_patches,
                                                                           self.k_neighbors).float()
            return k_nearest_patches_mask
        else:
            k_nearest_patches_mask = calculate_smaller_than_kth_value_mask(squared_distances, self.k_neighbors).float()
            if self.add_negative_patches_as_network_branch:
                k_nearest_negative_patches_mask = calculate_smaller_than_kth_value_mask(
                    squared_distances_to_negative_patches, self.k_neighbors).float()
                return k_nearest_patches_mask, k_nearest_negative_patches_mask
            else:
                return k_nearest_patches_mask


def sample_random_patches(data_loader,
                          n_patches: int,
                          patch_size: int,
                          func: Optional[Callable] = None,
                          visualize: bool = False):
    """
    This function sample random patches from the data, given by the data-loader object.
    It samples random indices for the patches and then iterates over the dataset to extract them.
    It returns a (NumPy) array containing the patches.
    """
    rng = np.random.default_rng()

    n_images, height, width, channels = data_loader.dataset.data.shape
    assert height == width, "Currently only square images are supported"
    spatial_size = height
    n_patches_per_row_or_col = spatial_size - patch_size + 1
    n_patches_per_image = n_patches_per_row_or_col ** 2
    n_patches_in_dataset = n_images * n_patches_per_image

    batch_size = data_loader.batch_size

    patches_indices_in_dataset = rng.choice(n_patches_in_dataset, size=n_patches, replace=False)

    images_indices = patches_indices_in_dataset % n_images
    patches_indices_in_images = patches_indices_in_dataset // n_images
    patches_x_indices_in_images = patches_indices_in_images % n_patches_per_row_or_col
    patches_y_indices_in_images = patches_indices_in_images // n_patches_per_row_or_col

    batches_indices = images_indices // batch_size
    images_indices_in_batches = images_indices % batch_size

    patches = np.empty(shape=(n_patches, channels, patch_size, patch_size), dtype=np.float32)

    for batch_index, (inputs, _) in enumerate(data_loader):
        if batch_index not in batches_indices:
            continue

        relevant_patches_mask = (batch_index == batches_indices)
        relevant_patches_indices = np.where(relevant_patches_mask)[0]

        if func is not None:
            inputs = func(inputs)
        inputs = inputs.cpu().numpy()

        for i in relevant_patches_indices:
            image_index_in_batch = images_indices_in_batches[i]
            patch_x_start = patches_x_indices_in_images[i]
            patch_y_start = patches_y_indices_in_images[i]
            patch_x_slice = slice(patch_x_start, patch_x_start + patch_size)
            patch_y_slice = slice(patch_y_start, patch_y_start + patch_size)

            patches[i] = inputs[image_index_in_batch, :, patch_x_slice, patch_y_slice]

            if visualize:
                visualize_image_patch_pair(image=inputs[image_index_in_batch], patch=patches[i],
                                           patch_x_start=patch_x_start, patch_y_start=patch_y_start)

    return patches


def visualize_image_patch_pair(image, patch, patch_x_start, patch_y_start):
    patch_size = patch.shape[0]
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


def train_model(model, dataloaders, num_epochs, device, criterion, optimizer, scheduler, log_interval):
    best_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    total_time = 0
    training_step = 0
    interval_accumulator = Accumulator()
    epoch_accumulator = Accumulator()

    for epoch in range(num_epochs):
        # model_state = copy.deepcopy(model.state_dict())
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

        # # For debugging purposes - verify that the weights of the model changed.
        # new_model_state = copy.deepcopy(model.state_dict())
        # for weight_name in new_model_state.keys():
        #     old_weight = model_state[weight_name]
        #     new_weight = new_model_state[weight_name]
        #     if torch.allclose(old_weight, new_weight):
        #         logger.warning(f'Weight \'{weight_name}\' of shape {list(new_weight.size())} did not change.')
        #     else:
        #         logger.debug(f'Weight \'{weight_name}\' of shape {list(new_weight.size())} changed.')
        # model_state = copy.deepcopy(new_model_state)

        epoch_test_loss, epoch_test_accuracy = evaluate_model(model, criterion, dataloaders['test'], device)
        wandb.log(data={'test_accuracy': epoch_test_accuracy, 'test_loss': epoch_test_loss}, step=training_step)

        # if the current model reached the best results so far, deep copy the weights of the model.
        if epoch_test_accuracy > best_accuracy:
            best_accuracy = epoch_test_accuracy
            best_weights = copy.deepcopy(model.state_dict())

        scheduler.step()

        epoch_time_elapsed = epoch_accumulator.get_time()
        total_time += epoch_time_elapsed
        epochs_left = num_epochs - (epoch + 1)
        avg_epoch_time = total_time / (epoch + 1)
        time_left = avg_epoch_time * epochs_left
        logger.info(f'Epoch {epoch + 1:0>3d}/{num_epochs:0>3d} '
                    f'({str(timedelta(seconds=epoch_time_elapsed)).split(".")[0]}) | '
                    f'ETA {str(timedelta(seconds=time_left)).split(".")[0]} | '
                    f'Train '
                    f'loss={epoch_accumulator.get_mean_loss():.4f} '
                    f'acc={epoch_accumulator.get_accuracy():.2f}% | '
                    f'Test '
                    f'loss={epoch_test_loss:.4f} '
                    f'acc={epoch_test_accuracy:.2f}%')

    logger.info(f'Best test accuracy: {best_accuracy:.2f}%')

    model.load_state_dict(best_weights)  # load best model weights
    return model


def calc_covariance(tensor, mean):
    centered_tensor = tensor - mean
    return (centered_tensor @ centered_tensor.t()) / tensor.size(1)


def calc_mean_patch(dataloader, patch_size, agg_func: Callable):
    total_size = 0
    mean = None
    for inputs, _ in dataloader:
        # if total_size > 200:   # Might be useful for debugging purposes...
        #     break

        # Unfold the input batch to its patches - shape (N, C*H*W, M) where M is the number of patches per image.
        patches = F.unfold(inputs, patch_size)

        # Replace the batch axis with the patch axis, to obtain shape (C*H*W, N, M)
        patches = patches.transpose(0, 1).contiguous()  # TODO is .contiguous() needed ?

        # Flatten the batch and n_patches axes, to obtain shape (C*H*W, N*M)
        patches = torch.flatten(patches, start_dim=1).double()

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


def calc_whitening(dataloader, patch_size, whitening_regularization_factor) -> np.ndarray:
    logger.info('Performing a first pass over the dataset to calculate the mean patch.')
    mean_patch = calc_mean_patch(dataloader, patch_size, agg_func=partial(torch.mean, dim=1))
    mean_as_column_vector = torch.unsqueeze(mean_patch, dim=-1)
    logger.info('Performing a second pass over the dataset to calculate the covariance.')
    covariance = calc_mean_patch(dataloader, patch_size, agg_func=partial(calc_covariance, mean=mean_as_column_vector))

    eigenvalues, eigenvectors = np.linalg.eigh(covariance.cpu().numpy())

    inv_sqrt_eigenvalues = np.diag(1. / np.sqrt(eigenvalues + whitening_regularization_factor))
    whitening_matrix = eigenvectors.dot(inv_sqrt_eigenvalues)
    whitening_matrix = whitening_matrix.astype(np.float32)

    return whitening_matrix


def get_whitening_matrix(dataloader, use_whitening: bool, patch_size: int, regularization_factor: float) -> np.ndarray:
    matrix_size = 3 * (patch_size ** 2)
    if use_whitening:
        filename = f'whitening_matrix_patch_size_{patch_size}_reg_{regularization_factor}.bin'
        if os.path.isfile(filename):
            logger.info(f'Reading whitening matrix from file {filename}.')
            whitening_matrix = np.fromfile(filename, dtype=np.float32).reshape(matrix_size,
                                                                               matrix_size)
        else:
            logger.info('Calculating whitening matrix.')
            whitening_matrix = calc_whitening(dataloader, patch_size, regularization_factor)
            whitening_matrix.tofile(filename)
            logger.info(f'Whitening matrix saved in file {filename}.')
    else:
        whitening_matrix = np.eye(matrix_size, dtype=np.float32)

    return whitening_matrix


def get_conv_kernel_and_bias(batch_size, n_patches, patch_size,
                             use_whitening: bool = False,
                             whitening_regularization_factor: float = 0.):
    clean_dataloaders = get_dataloaders(batch_size,
                                        normalize_to_unit_gaussian=False,
                                        normalize_to_plus_minus_one=False,
                                        random_crop=False,
                                        random_horizontal_flip=False,
                                        random_erasing=False,
                                        random_resized_crop=False)

    patches = sample_random_patches(clean_dataloaders['train'],
                                    n_patches=n_patches,
                                    patch_size=patch_size,
                                    visualize=False)

    whitening_matrix = get_whitening_matrix(clean_dataloaders['train'],
                                            use_whitening, patch_size, whitening_regularization_factor)

    patches_flattened = patches.reshape(patches.shape[0], -1)
    kernel = np.linalg.multi_dot([patches_flattened, whitening_matrix, whitening_matrix.T])
    bias = np.linalg.norm(patches_flattened.dot(whitening_matrix), axis=1) ** 2

    kernel = torch.from_numpy(kernel).view(patches.shape).float()
    bias = torch.from_numpy(bias).float()

    return kernel, bias


def main():
    args = parse_args()
    out_dir = create_out_dir(args.path)
    configure_logger(out_dir)

    logger.info(f'Starting to train patch-based-classifier '
                f'for {args.epochs} epochs '
                f'(using device {args.device})')
    logger.info(f'batch_size={args.batch_size}, '
                f'learning_rate={args.learning_rate}, '
                f'weight_decay={args.weight_decay}')
    logger.info(f'n_patches={args.n_patches}, '
                f'patch_size={args.patch_size}')
    if args.use_avg_pool:
        logger.info(f'Using AvgPool (size={args.pool_size}, stride={args.pool_stride})')
    if args.use_whitening:
        logger.info(f'Using whitening (whitening_regularization_factor={args.whitening_regularization_factor}')

    kernel, bias = get_conv_kernel_and_bias(args.batch_size, args.n_patches, args.patch_size,
                                            args.use_whitening, args.whitening_regularization_factor)
    model = ClassifierOnPatchBasedEmbedding(
        kernel_convolution=kernel, bias_convolution=bias,
        k_neighbors=int(args.k_neighbors_fraction * args.n_patches),
        n_channels=args.n_channels, add_flipped_patches=args.add_flipped_patches,
        add_negative_patches_as_network_branch=args.add_negative_patches_as_network_branch,
        add_negative_patches_as_more_patches=args.add_negative_patches_as_more_patches,
        use_batch_norm=args.use_batch_norm,
        conv_kernel_size=args.conv_kernel_size,
        use_avg_pool=args.use_avg_pool, 
        pool_size=args.pool_size,
        pool_stride=args.pool_stride,
        use_adaptive_avg_pool=args.use_adaptive_avg_pool,
        use_relu_after_bottleneck=args.use_relu_after_bottleneck
    )
    device = torch.device(args.device)
    model = model.to(device)

    wandb.init(project='thesis', config=args)
    wandb.watch(model)

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     milestones=args.learning_rate_decay_steps,
                                                     gamma=args.learning_rate_decay_gamma)
    augmented_dataloaders = get_dataloaders(args.batch_size,
                                            normalize_to_unit_gaussian=args.enable_normalization_to_unit_gaussian,
                                            normalize_to_plus_minus_one=not args.disable_normalization_to_plus_minus_one,
                                            random_crop=not args.disable_random_crop,  # TODO change to True
                                            random_horizontal_flip=not args.disable_random_horizontal_flip,
                                            random_erasing=args.enable_random_erasing,
                                            random_resized_crop=args.enable_random_resized_crop)

    train_model(model, augmented_dataloaders, args.epochs, device, criterion, optimizer, scheduler, args.log_interval)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Main script for running the experiments for patches-based learning.'
                    'The experiments results are outputted to a log-file and to wandb.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Arguments defining the model.
    parser.add_argument('--n_channels', type=int, default=128,
                        help=f'Number of channels in the convolution layer which comes after the embedding')

    # Arguments defining the training-process
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu'] + [f'cuda:{i}' for i in range(8)],
                        help=f'On which device to train')
    parser.add_argument('--epochs', type=int, default=80,
                        help=f'Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help=f'Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help=f'Learning-rate')
    parser.add_argument('--learning_rate_decay_steps', type=int, nargs='+', default=[50, 75],
                        help=f'Decay the leraning-rate at these steps by a factor of gamma '
                             f'(given as another argument)')
    parser.add_argument('--learning_rate_decay_gamma', type=float, default=0.1,
                        help=f'The factor gamma to multiply the learning-rate at the decay steps')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help=f'Momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help=f'Weight decay')

    parser.add_argument('--n_patches', type=int, default=2048,
                        help=f'The number of patches')
    parser.add_argument('--patch_size', type=int, default=6,
                        help=f'The size of the patches')
    parser.add_argument('--use_avg_pool', action='store_true',
                        help='If true, use whitening on the patches')
    parser.add_argument('--conv_kernel_size', type=int, default=1,
                        help=f'The size of the kernel in the convolution layer after the patch-based-embedding '
                             f'(a.k.a. \"bottle-neck\" layer)')
    parser.add_argument('--pool_size', type=int, default=5,
                        help=f'The size of the average-pooling layer after the patch-based-embedding')
    parser.add_argument('--pool_stride', type=int, default=3,
                        help=f'The stride of the average-pooling layer after the patch-based-embedding')
    parser.add_argument('--k_neighbors_fraction', type=float, default=0.4,
                        help=f'which k to use in the k-nearest-neighbors, as a fraction of the total number of patches')
    parser.add_argument('--use_whitening', action='store_true',
                        help='If true, use whitening on the patches')
    parser.add_argument('--whitening_regularization_factor', type=float, default=0.001,
                        help=f'The regularization factor (a.k.a. lambda) of the whitening matrix')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help=f'Whether to use batch normalization after the patch-based-embedding')
    parser.add_argument('--use_relu_after_bottleneck', action='store_true',
                        help=f'Whether to use ReLU after the bottleneck layer')
    parser.add_argument('--add_negative_patches_as_network_branch', action='store_true',
                        help=f'Whether to use the negative patches as well (i.e. original patches multiplied by -1). '
                             f'These patches are being used as a separate network branch, as the original paper.')
    parser.add_argument('--add_negative_patches_as_more_patches', action='store_true',
                        help=f'Whether to use the negative patches as well (i.e. original patches multiplied by -1). '
                             f'These patches are being concatenated to the original patches so the dictionary size is '
                             f'multiplied by two')
    parser.add_argument('--add_flipped_patches', action='store_true',
                        help=f'Whether to use the negative patches as well (i.e. original patches multiplied by -1)')
    parser.add_argument('--use_adaptive_avg_pool', action='store_true',
                        help=f'Whether to use the adaptive avg-pooling on the embedding output to get spatial size 6')

    # Arguments for logging the training process.
    parser.add_argument('--path', type=str, default='./experiments',
                        help=f'Output path for the experiment - '
                             f'a sub-directory named with the data and time will be created within')
    parser.add_argument('--log_interval', type=int, default=100,
                        help=f'How many iterations between each training log')

    # Arguments for the data augmentations.
    # These refer to data augmentations that are enabled by default (that's way they are prefixed with 'disable').
    parser.add_argument('--disable_normalization_to_plus_minus_one', action='store_true',
                        help='If true, disable normalization of the values to the range [-1,1] (instead of [0,1])')
    parser.add_argument('--disable_random_crop', action='store_true',
                        help='If true, disable random cropping which is padding of 4 followed by random crop')
    parser.add_argument('--disable_random_horizontal_flip', action='store_true',
                        help='If true, disable random horizontal flip')
    # These refer to data augmentations that can be enabled (that's way they are prefixed with 'enable').
    parser.add_argument('--enable_random_resized_crop', action='store_true',
                        help='If true, enable random resized cropping')
    parser.add_argument('--enable_normalization_to_unit_gaussian', action='store_true',
                        help='If true, enable normalization of the values to a unit gaussian')
    parser.add_argument('--enable_random_erasing', action='store_true',
                        help='If true, performs erase a random rectangle in the image')

    return parser.parse_args()


if __name__ == '__main__':
    main()
