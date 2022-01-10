"""
This file runs the experiments for patches-based learning, similar to
The Unreasonable Effectiveness of Patches in Deep Convolutional Kernels Methods
(https://arxiv.org/pdf/2101.07528.pdf)
"""
import copy
import math

import wandb
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from functools import partial
from math import ceil
from torchvision.transforms.functional import hflip
from matplotlib.patches import Rectangle
from loguru import logger
from typing import Callable, Optional, Tuple
from datetime import timedelta

from consts import CIFAR10_IMAGE_SIZE, N_CLASSES
from schemas.patches import Args, ArchitectureArgs
from utils import (configure_logger,
                   get_dataloaders,
                   Accumulator,
                   perform_train_step_regular,
                   evaluate_model, 
                   get_args, 
                   log_args, 
                   get_model_device, 
                   get_model_output_shape)

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
def sample_random_patches(data_loader,
                          n_patches,
                          patch_size,
                          existing_model: Optional[nn.Module] = None,
                          visualize: bool = False,
                          random_uniform_patches: bool = False,
                          random_gaussian_patches: bool = False):
    """
    This function sample random patches from the data, given by the data-loader object.
    It samples random indices for the patches and then iterates over the dataset to extract them.
    It returns a (NumPy) array containing the patches.
    """
    batch_size = data_loader.batch_size
    n_images = data_loader.dataset.data.shape[0]
    patch_shape = data_loader.dataset.data.shape[1:]
    patch_shape = np.roll(patch_shape, shift=1)  # In the dataset it's H x W x C but in the model it's C x H x W
    if existing_model is not None:
        device = get_model_device(existing_model)
        patch_shape = get_model_output_shape(existing_model)

    if len(patch_shape) > 1:
        assert len(patch_shape) == 3 and (patch_shape[1] == patch_shape[2]), "Should be C x H x W where H = W"
        spatial_size = patch_shape[-1]
        if patch_size == -1:  # -1 means the patch size is the whole size of the image (or down-sampled activation)
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

    patches = np.empty(shape=(n_patches, ) + patch_shape, dtype=np.float32)

    if random_uniform_patches:
        return np.random.default_rng().uniform(low=-1, high=+1, size=patches.shape).astype(np.float32)
    if random_gaussian_patches:
        patch_dim = math.prod(patch_shape)
        return np.random.default_rng().multivariate_normal(
            mean=np.zeros(patch_dim), cov=np.eye(patch_dim), size=n_patches).astype(np.float32).reshape(patches.shape)

    for batch_index, (inputs, _) in enumerate(data_loader):
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


def unwhiten_patches(patches: np.ndarray, args: Args) -> np.ndarray:
    assert args.runtime.wwt_inv is not None, \
        "How did we reach the function 'unwhiten_patches' if the whitening matrix wasn't calculated yet?"

    patches_flattened = patches.reshape(patches.shape[0], -1)
    patches_orig_flattened = np.dot(patches_flattened, args.runtime.wwt_inv)
    patches_orig = patches_orig_flattened.reshape(patches.shape)

    return patches_orig


def get_extreme_patches_indices(norms_numpy, number_of_extreme_patches_to_show):
    partitioned_indices = np.argpartition(norms_numpy, number_of_extreme_patches_to_show)
    worst_patches_indices = partitioned_indices[:number_of_extreme_patches_to_show]
    partitioned_indices = np.argpartition(norms_numpy, len(norms_numpy) - number_of_extreme_patches_to_show)
    best_patches_indices = partitioned_indices[-number_of_extreme_patches_to_show:]

    return worst_patches_indices, best_patches_indices


def get_extreme_patches_unwhitened(model, args, worst_patches_indices, best_patches_indices):
    all_patches = model.patch_based_embedding.kernel_convolution.cpu().numpy()
    worst_patches = all_patches[worst_patches_indices]
    best_patches = all_patches[best_patches_indices]

    both_patches = np.concatenate([worst_patches, best_patches])
    both_patches_unwhitened = unwhiten_patches(both_patches, args)

    worst_patches_unwhitened = both_patches_unwhitened[:len(worst_patches)]
    best_patches_unwhitened = both_patches_unwhitened[len(best_patches):]

    return worst_patches_unwhitened, best_patches_unwhitened


def visualize_patches(model: ClassifierOnPatchBasedEmbedding, args: Args, n: int = 3):
    bottleneck_weight = model.bottle_neck_conv_1.weight.data.squeeze(dim=3).squeeze(dim=2)
    for norm_ord in [1, 2, np.inf]:
        norms = torch.linalg.norm(bottleneck_weight, ord=norm_ord, dim=0)
        norms_numpy = norms.cpu().numpy()
        wandb.log({f'L{norm_ord}_norm_patches_weights': wandb.Histogram(norms_numpy)}, step=training_step)

        worst_patches_indices, best_patches_indices = get_extreme_patches_indices(norms_numpy, n ** 2)
        worst_patches_unwhitened, best_patches_unwhitened = get_extreme_patches_unwhitened(
            model, args, worst_patches_indices, best_patches_indices)

        # Due to numerical issues sometimes the values are slightly outside [0,1] which causes annoying plt warning
        worst_patches_unwhitened = np.clip(worst_patches_unwhitened, 0, 1)
        best_patches_unwhitened = np.clip(best_patches_unwhitened, 0, 1)

        worst_patches_fig, worst_patches_axs = plt.subplots(n, n)
        best_patches_fig, best_patches_axs = plt.subplots(n, n)

        best_patches_fig.suptitle('Best patches (i.e. high norm)')
        worst_patches_fig.suptitle('Worst patches (i.e. low norm)')

        for i in range(n ** 2):
            row_index = i // n
            col_index = i % n
            best_patches_axs[row_index, col_index].imshow(best_patches_unwhitened[i].transpose(1, 2, 0),
                                                          vmin=0, vmax=1)
            worst_patches_axs[row_index, col_index].imshow(worst_patches_unwhitened[i].transpose(1, 2, 0),
                                                           vmin=0, vmax=1)
            best_patches_axs[row_index, col_index].axis('off')
            worst_patches_axs[row_index, col_index].axis('off')

        wandb.log({'best_patches': best_patches_fig, 'worst_patches': worst_patches_fig}, step=training_step)
        plt.close('all')  # Avoid memory consumption


@torch.no_grad()
def kill_weak_patches(model: ClassifierOnPatchBasedEmbedding, args: Args) -> ClassifierOnPatchBasedEmbedding:
    bottleneck_weight = model.bottle_neck_conv_1.weight.data.squeeze(dim=3).squeeze(dim=2)
    norms = torch.linalg.norm(bottleneck_weight, ord=1, dim=0)
    quantile = torch.quantile(norms, 1 - args.arch.survival_of_the_fittest_fraction_of_survivals)
    strong_patches_mask = torch.greater(norms, quantile)
    weak_patches_mask = torch.logical_not(strong_patches_mask)
    n_weak_patches = torch.sum(weak_patches_mask).cpu().numpy().item()
    kernel, bias = get_conv_kernel_and_bias(args)  # TODO existing model=?

    model.patch_based_embedding.kernel_convolution[weak_patches_mask, :, :, :] = kernel.to(args.env.device)
    model.patch_based_embedding.bias_convolution[:, weak_patches_mask, :, :] = bias.view(
        (1, n_weak_patches, 1, 1)).to(args.env.device)

    return model


training_step = 0


def train_model(args: Args,
                model: ClassifierOnPatchBasedEmbedding,
                dataloaders,
                criterion: nn.CrossEntropyLoss,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.MultiStepLR,
                inputs_preprocessing_function: Optional[Callable] = None):
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

        if (args.arch.survival_of_the_fittest_enabled and
                (epoch != 0) and
                (epoch % args.arch.survival_of_the_fittest_rate_of_evolution_in_epochs == 0)):
            model = kill_weak_patches(model, args)

        for inputs, labels in dataloaders['train']:
            # if np.random.binomial(n=1, p=0.1) == 1:  # TODO temporary
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
                visualize_patches(model, args)

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

        epoch_test_loss, epoch_test_accuracy = evaluate_model(model, criterion, dataloaders['test'], args.env.device,
                                                              inputs_preprocessing_function)
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


def get_conv_kernel_and_bias(args: Args,
                             existing_model: Optional[nn.Module] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    clean_dataloaders = get_dataloaders(args.opt.batch_size,
                                        normalize_to_unit_gaussian=args.data.normalization_to_unit_gaussian,
                                        normalize_to_plus_minus_one=args.data.normalization_to_plus_minus_one)

    patches = sample_random_patches(clean_dataloaders['train'],
                                    args.arch.n_patches,
                                    args.arch.patch_size,
                                    existing_model)
    patches_flattened = patches.reshape(patches.shape[0], -1)

    if args.arch.use_whitening:
        if args.runtime.whitening_matrix is None:
            args.runtime.whitening_matrix = calc_whitening(args, clean_dataloaders['train'], existing_model)
            args.runtime.wwt = np.dot(args.runtime.whitening_matrix, np.transpose(args.runtime.whitening_matrix))
            args.runtime.wwt_inv = np.linalg.inv(args.runtime.wwt)  # Might be used later for un-whitening

        kernel = np.dot(patches_flattened, args.runtime.wwt).reshape(patches.shape)
        bias = np.linalg.norm(patches_flattened.dot(args.runtime.whitening_matrix), axis=1) ** 2
    else:
        kernel = patches
        bias = np.linalg.norm(patches_flattened, axis=1) ** 2

    kernel = torch.from_numpy(kernel).float()
    bias = torch.from_numpy(bias).float()

    return kernel, bias


def train_patch_based_model(args: Args, existing_model: Optional[nn.Module] = None):
    device = torch.device(args.env.device)
    _, height, width = (3, 32, 32) if (existing_model is None) else get_model_output_shape(existing_model)
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


def main():
    args = get_args(args_class=Args)

    configure_logger(args.env.path)
    log_args(args)

    model = train_patch_based_model(args)

    if args.arch.depth == 2:
        model.eval()
        model.prediction_mode_off()
        # TODO remove gradients from the model computational-graph since they are no longer needed.
        model2 = train_patch_based_model(args, existing_model=model)


if __name__ == '__main__':
    main()
