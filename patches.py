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
from functools import partial
from math import ceil

import yaml
from torchvision.transforms.functional import hflip

import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from loguru import logger
from typing import Callable, Optional, Tuple
from datetime import timedelta

from consts import CIFAR10_IMAGE_SIZE, N_CLASSES
from schemas.patches import Args, ArchitectureArgs
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
    return torch.less(x, x.kthvalue(dim=1, k=k + 1, keepdim=True).values)


class ClassifierOnPatchBasedEmbedding(nn.Module):
    def __init__(self,
                 args: ArchitectureArgs,
                 kernel_convolution: torch.Tensor,
                 bias_convolution: torch.Tensor,
                 input_image_spatial_size: int = CIFAR10_IMAGE_SIZE):
        super(ClassifierOnPatchBasedEmbedding, self).__init__()

        self.args = args
        self.embedding_n_channels = (2 ** (int(args.add_negative_patches_as_more_patches) +
                                           int(args.add_flipped_patches))) * args.n_patches

        embedding_spatial_size = input_image_spatial_size - args.patch_size + 1
        pooled_embedding_dim = ceil((embedding_spatial_size - args.pool_size) / args.pool_stride + 1)
        conv_input_spatial_size = pooled_embedding_dim if args.use_avg_pool else embedding_spatial_size
        if args.use_adaptive_avg_pool:
            conv_input_spatial_size = 6
        conv_output_spatial_size = conv_input_spatial_size - args.conv_kernel_size + 1
        intermediate_n_features = args.n_channels * (conv_output_spatial_size ** 2)

        self.patch_based_embedding = PatchBasedEmbedding(args, kernel_convolution, bias_convolution)
        self.avg_pool = nn.AvgPool2d(args.pool_size, args.pool_stride, ceil_mode=True) if args.use_avg_pool else None
        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=6) if args.use_adaptive_avg_pool else None
        self.flatten = nn.Flatten()
        self.bottleneck_relu = nn.ReLU() if args.use_relu_after_bottleneck else None
        self.final_layer = nn.Linear(in_features=intermediate_n_features, out_features=N_CLASSES)

        if not args.add_negative_patches_as_network_branch:
            self.batch_norm = nn.BatchNorm2d(self.embedding_n_channels) if args.use_batch_norm else None
            self.bottle_neck_conv = nn.Conv2d(in_channels=self.embedding_n_channels,
                                              out_channels=args.n_channels,
                                              kernel_size=args.conv_kernel_size)
        else:
            self.batch_norm_1 = nn.BatchNorm2d(args.n_patches) if args.use_batch_norm else None
            self.batch_norm_2 = nn.BatchNorm2d(args.n_patches) if args.use_batch_norm else None
            self.bottle_neck_conv_1 = nn.Conv2d(in_channels=args.n_patches,
                                                out_channels=args.n_channels,
                                                kernel_size=args.conv_kernel_size)
            self.bottle_neck_conv_2 = nn.Conv2d(in_channels=args.n_patches,
                                                out_channels=args.n_channels,
                                                kernel_size=args.conv_kernel_size)

        self.scores_prediction_mode: bool = True

    def prediction_mode_off(self):
        self.scores_prediction_mode = False

    def prediction_mode_on(self):
        self.scores_prediction_mode = True

    def forward(self, x):
        if not self.args.add_negative_patches_as_network_branch:
            embedding = self.patch_based_embedding(x)
            if self.args.use_avg_pool:
                embedding = self.avg_pool(embedding)
            if self.args.use_adaptive_avg_pool:
                embedding = self.adaptive_avg_pool(embedding)
            if self.args.use_batch_norm:
                embedding = self.batch_norm(embedding)
            embedding = self.bottle_neck_conv(embedding)
        else:
            embedding1, embedding2 = self.patch_based_embedding(x)

            if self.args.use_avg_pool:
                embedding1 = self.avg_pool(embedding1)
                embedding2 = self.avg_pool(embedding2)

            if self.args.use_adaptive_avg_pool:
                embedding1 = self.adaptive_avg_pool(embedding1)
                embedding2 = self.adaptive_avg_pool(embedding2)

            if self.args.use_batch_norm:
                embedding1 = self.batch_norm_1(embedding1)
                embedding2 = self.batch_norm_2(embedding2)

            embedding1 = self.bottle_neck_conv_1(embedding1)
            embedding2 = self.bottle_neck_conv_2(embedding2)

            # TODO when add_negative_patches_as_network_branch is True, try to perform ReLU on each separate embedding
            # if self.bottleneck_relu:
            #     embedding1 = self.bottleneck_relu(embedding1)
            #     embedding2 = self.bottleneck_relu(embedding2)

            embedding = embedding1 + embedding2

        if self.scores_prediction_mode:
            if self.args.use_relu_after_bottleneck:
                embedding = self.bottleneck_relu(embedding)
            embedding_flat = self.flatten(embedding)
            scores = self.final_layer(embedding_flat)
            return scores
        else:
            return embedding


class PatchBasedEmbedding(nn.Module):
    """
    Calculating the k-nearest-neighbors is implemented as convolution with bias, as was done in
    The Unreasonable Effectiveness of Patches in Deep Convolutional Kernels Methods
    (https://arxiv.org/pdf/2101.07528.pdf)
    Details can be found in Appendix B (page 13).
    """

    def __init__(self,
                 args: ArchitectureArgs,
                 kernel_convolution: torch.Tensor,
                 bias_convolution: torch.Tensor):
        super(PatchBasedEmbedding, self).__init__()

        self.args = args
        self.kernel_convolution = nn.Parameter(kernel_convolution, requires_grad=False)

        # The bias will be added to a tensor of shape (N, n_patches, H, W) to reshaping it to (1, n_patches, 1, 1)
        # will make the addition "broadcastable".
        self.bias_convolution = nn.Parameter(bias_convolution.view(1, -1, 1, 1), requires_grad=False)

        if args.add_flipped_patches:
            self.flipped_kernel = nn.Parameter(hflip(self.kernel_convolution), requires_grad=False)
        else:
            self.flipped_kernel = None

    def forward(self, images):
        conv_result_no_bias = F.conv2d(images, self.kernel_convolution)
        squared_distances = -1 * conv_result_no_bias + self.bias_convolution

        calc_negative_patches = (self.args.add_negative_patches_as_network_branch or
                                 self.args.add_negative_patches_as_more_patches)
        squared_distances_to_negative_patches = (conv_result_no_bias + self.bias_convolution if calc_negative_patches
                                                 else None)

        all_squared_distances = [squared_distances]

        if self.args.add_flipped_patches:
            conv_result_to_flipped_kernel_no_bias = F.conv2d(images, self.flipped_kernel)
            squared_distances_to_flipped_patches = -1 * conv_result_to_flipped_kernel_no_bias + self.bias_convolution
            all_squared_distances.append(squared_distances_to_flipped_patches)

        if self.args.add_negative_patches_as_more_patches:
            all_squared_distances.append(squared_distances_to_negative_patches)
            if self.args.add_flipped_patches:
                squared_distances_to_flipped_negative_patches = (conv_result_to_flipped_kernel_no_bias +
                                                                 self.bias_convolution)
                all_squared_distances.append(squared_distances_to_flipped_negative_patches)

        if self.args.add_flipped_patches or self.args.add_negative_patches_as_more_patches:
            squared_distances_to_all_patches = torch.cat(all_squared_distances, dim=1)
            k_nearest_patches_mask = calculate_smaller_than_kth_value_mask(squared_distances_to_all_patches,
                                                                           self.args.k_neighbors).float()
            return k_nearest_patches_mask
        else:
            k_nearest_patches_mask = calculate_smaller_than_kth_value_mask(squared_distances,
                                                                           self.args.k_neighbors).float()
            if self.args.add_negative_patches_as_network_branch:
                k_nearest_negative_patches_mask = calculate_smaller_than_kth_value_mask(
                    squared_distances_to_negative_patches, self.args.k_neighbors).float()
                return k_nearest_patches_mask, k_nearest_negative_patches_mask
            else:
                return k_nearest_patches_mask


@torch.no_grad()
def sample_random_patches(args: Args,
                          data_loader,
                          existing_model: Optional[nn.Module] = None,
                          visualize: bool = False):
    """
    This function sample random patches from the data, given by the data-loader object.
    It samples random indices for the patches and then iterates over the dataset to extract them.
    It returns a (NumPy) array containing the patches.
    """
    rng = np.random.default_rng()

    batch_size = data_loader.batch_size
    n_images, height, width, channels = data_loader.dataset.data.shape
    if existing_model is not None:
        channels, height, width = get_model_output_shape(existing_model, args.env.device)

    spatial_size = height
    n_patches_per_row_or_col = spatial_size - args.arch.patch_size + 1
    n_patches_per_image = n_patches_per_row_or_col ** 2
    n_patches_in_dataset = n_images * n_patches_per_image

    patches_indices_in_dataset = np.sort(rng.choice(n_patches_in_dataset, size=args.arch.n_patches, replace=False))

    images_indices = patches_indices_in_dataset % n_images
    patches_indices_in_images = patches_indices_in_dataset // n_images
    patches_x_indices_in_images = patches_indices_in_images % n_patches_per_row_or_col
    patches_y_indices_in_images = patches_indices_in_images // n_patches_per_row_or_col

    batches_indices = images_indices // batch_size
    images_indices_in_batches = images_indices % batch_size

    patches = np.empty(shape=(args.arch.n_patches, channels, args.arch.patch_size, args.arch.patch_size),
                       dtype=np.float32)

    for batch_index, (inputs, _) in enumerate(data_loader):
        if batch_index not in batches_indices:
            continue

        relevant_patches_mask = (batch_index == batches_indices)
        relevant_patches_indices = np.where(relevant_patches_mask)[0]

        if existing_model is not None:
            inputs = inputs.to(args.env.device)
            inputs = existing_model(inputs)
        inputs = inputs.cpu().numpy()

        for i in relevant_patches_indices:
            image_index_in_batch = images_indices_in_batches[i]
            patch_x_start = patches_x_indices_in_images[i]
            patch_y_start = patches_y_indices_in_images[i]
            patch_x_slice = slice(patch_x_start, patch_x_start + args.arch.patch_size)
            patch_y_slice = slice(patch_y_start, patch_y_start + args.arch.patch_size)

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

training_step = 0

def train_model(args: Args, model, dataloaders, criterion, optimizer, scheduler, inputs_preprocessing_function=None):
    best_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    global training_step
    total_time = 0
    interval_accumulator = Accumulator()
    epoch_accumulator = Accumulator()

    for epoch in range(args.opt.epochs):
        # if epoch > 0:  # TODO temporary
        #     break
        # model_state = copy.deepcopy(model.state_dict())
        model.train()
        epoch_accumulator.reset()

        for inputs, labels in dataloaders['train']:
            # if training_step > 1:  # TODO temporary
            #     break
            training_step += 1

            inputs = inputs.to(args.env.device)
            labels = labels.to(args.env.device)

            if inputs_preprocessing_function is not None:
                with torch.no_grad():
                    inputs = inputs_preprocessing_function(inputs)
                inputs = inputs.detach()  # TODO is it needed?

            loss, predictions = perform_train_step_regular(model, inputs, labels, criterion, optimizer)

            accumulator_kwargs = dict(mean_loss=loss,
                                      num_corrects=torch.sum(torch.eq(predictions, labels.data)).item(),
                                      n_samples=inputs.size(0))  # This equals the batch-size, except in the last batch

            epoch_accumulator.update(**accumulator_kwargs)
            interval_accumulator.update(**accumulator_kwargs)

            if training_step % args.env.log_interval == 0:
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

        epoch_test_loss, epoch_test_accuracy = evaluate_model(model, criterion, dataloaders['test'], args.env.device, inputs_preprocessing_function)
        # epoch_test_loss, epoch_test_accuracy = 0.5, 90  # TODO temporary
        wandb.log(data={'test_accuracy': epoch_test_accuracy, 'test_loss': epoch_test_loss}, step=training_step)

        # if the current model reached the best results so far, deep copy the weights of the model.
        if epoch_test_accuracy > best_accuracy:
            best_accuracy = epoch_test_accuracy
            best_weights = copy.deepcopy(model.state_dict())

        scheduler.step()

        epoch_time_elapsed = epoch_accumulator.get_time()
        total_time += epoch_time_elapsed
        epochs_left = args.opt.epochs - (epoch + 1)
        avg_epoch_time = total_time / (epoch + 1)
        time_left = avg_epoch_time * epochs_left
        logger.info(f'Epoch {epoch + 1:0>3d}/{args.opt.epochs:0>3d} '
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


@torch.no_grad()
def calc_mean_patch(dataloader,
                    patch_size,
                    agg_func: Callable,
                    existing_model: Optional[nn.Module] = None,
                    device: Optional[torch.device] = None):
    total_size = 0
    mean = None
    for inputs, _ in dataloader:
        # if total_size > 200:  # TODO temporary
        #     break

        if existing_model is not None:
            inputs = inputs.to(device)
            inputs = existing_model(inputs)
            inputs = inputs.cpu()

        # Unfold the input batch to its patches - shape (N, C*H*W, M) where M is the number of patches per image.
        patches = F.unfold(inputs, patch_size)

        # Replace the batch axis with the patch axis, to obtain shape (C*H*W, N, M)
        patches = patches.transpose(0, 1).contiguous()

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


def calc_whitening(args: Args, dataloader, existing_model: Optional[nn.Module] = None) -> np.ndarray:
    logger.info('Performing a first pass over the dataset to calculate the mean patch.')
    mean_patch = calc_mean_patch(dataloader, args.arch.patch_size,
                                 agg_func=partial(torch.mean, dim=1),
                                 existing_model=existing_model, device=args.env.device)
    logger.info('Performing a second pass over the dataset to calculate the covariance.')
    covariance = calc_mean_patch(dataloader, args.arch.patch_size,
                                 agg_func=partial(calc_covariance, mean=torch.unsqueeze(mean_patch, dim=-1)),
                                 existing_model=existing_model, device=args.env.device)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance.cpu().numpy())

    inv_sqrt_eigenvalues = np.diag(1. / np.sqrt(eigenvalues + args.arch.whitening_regularization_factor))
    whitening_matrix = eigenvectors.dot(inv_sqrt_eigenvalues)
    whitening_matrix = whitening_matrix.astype(np.float32)

    return whitening_matrix


def get_conv_kernel_and_bias(args: Args,
                             existing_model: Optional[nn.Module] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    clean_dataloaders = get_dataloaders(args.opt.batch_size,
                                        normalize_to_unit_gaussian=False,
                                        normalize_to_plus_minus_one=False,
                                        random_crop=False,
                                        random_horizontal_flip=False,
                                        random_erasing=False,
                                        random_resized_crop=False)

    patches = sample_random_patches(args, clean_dataloaders['train'], existing_model, visualize=False)
    patches_flattened = patches.reshape(patches.shape[0], -1)

    if args.arch.use_whitening:
        whitening_matrix = calc_whitening(args, clean_dataloaders['train'], existing_model)

        kernel = np.linalg.multi_dot([patches_flattened, whitening_matrix, whitening_matrix.T]).reshape(patches.shape)
        bias = np.linalg.norm(patches_flattened.dot(whitening_matrix), axis=1) ** 2
    else:
        kernel = patches
        bias = np.linalg.norm(patches_flattened, axis=1) ** 2

    kernel = torch.from_numpy(kernel).float()
    bias = torch.from_numpy(bias).float()

    return kernel, bias


@torch.no_grad()
def get_model_output_shape(model: nn.Module, device: torch.device):
    clean_dataloaders = get_dataloaders(batch_size=1,
                                        normalize_to_unit_gaussian=False,
                                        normalize_to_plus_minus_one=False,
                                        random_crop=False,
                                        random_horizontal_flip=False,
                                        random_erasing=False,
                                        random_resized_crop=False)
    inputs, _ = next(iter(clean_dataloaders["train"]))
    inputs = inputs.to(device)
    outputs = model(inputs)
    outputs = outputs.cpu().numpy()
    _, channels, height, width = outputs.shape
    return channels, height, width


def train_patch_based_model(args: Args, existing_model: Optional[nn.Module] = None):
    device = torch.device(args.env.device)
    _, height, width = (3, 32, 32) if (existing_model is None) else get_model_output_shape(existing_model, device)
    input_image_spatial_size = height

    kernel, bias = get_conv_kernel_and_bias(args, existing_model)

    model = ClassifierOnPatchBasedEmbedding(args.arch, kernel, bias, input_image_spatial_size).to(device)

    # Initialize the args to wandb without the category prefix
    wandb.init(project='thesis', config=args.flattened_dict())
    wandb.watch(model)

    optimizer = torch.optim.SGD(model.parameters(), args.opt.learning_rate, args.opt.momentum,
                                weight_decay=args.opt.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args.opt.learning_rate_decay_steps,
                                                     gamma=args.opt.learning_rate_decay_gamma)
    augmented_dataloaders = get_dataloaders(
        args.opt.batch_size,
        normalize_to_unit_gaussian=args.data.normalization_to_unit_gaussian,
        normalize_to_plus_minus_one=args.data.normalization_to_plus_minus_one,
        random_crop=args.data.random_crop,
        random_horizontal_flip=args.data.random_horizontal_flip
    )

    best_model = train_model(args, model, augmented_dataloaders, criterion, optimizer, scheduler,
                             inputs_preprocessing_function=existing_model)

    return best_model


def get_args() -> Args:
    known_args, unknown_args = parse_args()
    with open(known_args.yaml_path, 'r') as f:
        args_dict = yaml.load(f, Loader=yaml.FullLoader)

    while len(unknown_args) > 0:
        arg_name = unknown_args.pop(0).replace('--', '')
        values = list()
        while (len(unknown_args) > 0) and (not unknown_args[0].startswith('--')):
            values.append(unknown_args.pop(0))
        if len(values) == 0:
            raise ValueError(f'Argument {arg_name} given in command line has no corresponding value.')
        value = values[0] if len(values) == 1 else values

        categories = list(Args.__fields__.keys())
        found = False
        for category in categories:
            category_args = list(Args.__fields__[category].default.__fields__.keys())
            if arg_name in category_args:
                if category not in args_dict:
                    args_dict[category] = dict()
                args_dict[category][arg_name] = value
                found = True

        if not found:
            raise ValueError(f'Argument {arg_name} is not recognized.')

    args = Args.parse_obj(args_dict)

    return args


def log_args(args):
    logger.info(f'Starting to train patch-based-classifier with the following arguments:')
    for arg_name, value in args.flattened_dict().items():
        logger.info(f'{f"{arg_name} ":-<50} {value}')


def main():
    args = get_args()

    configure_logger(args.env.path)
    log_args(args)
    
    model = train_patch_based_model(args)

    if args.arch.depth == 2:
        model.eval()
        model.prediction_mode_off()
        # TODO remove gradients from the model computational-graph since they are no longer needed.
        model2 = train_patch_based_model(args, existing_model=model)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Main script for running the experiments for patches-based learning.'
                    'The experiments results are outputted to a log-file and to wandb.'
    )

    parser.add_argument('yaml_path', help=f'Path to a YAML file with the arguments according to the pydantic schema')

    return parser.parse_known_args()


if __name__ == '__main__':
    main()
