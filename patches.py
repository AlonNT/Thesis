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


def calculate_smaller_than_kth_value_mask(x, k) -> torch.Tensor:
    return torch.less(x, x.kthvalue(dim=1, k=k + 1, keepdim=True).values)


class ClassifierOnPatchBasedEmbedding(nn.Module):
    def __init__(self,
                 kernel_convolution,
                 bias_convolution,
                 pool_size,
                 pool_stride,
                 k_neighbors,
                 n_channels):
        super(ClassifierOnPatchBasedEmbedding, self).__init__()

        self.n_patches = kernel_convolution.shape[0]
        self.patch_based_embedding = PatchBasedEmbedding(kernel_convolution,
                                                         bias_convolution,
                                                         pool_size,
                                                         pool_stride,
                                                         k_neighbors)
        kernel_size = kernel_convolution.size(-1)
        embedding_spatial_size = CIFAR10_IMAGE_SIZE - kernel_size + 1
        intermediate_n_features = n_channels * (embedding_spatial_size ** 2)

        self.classifier = nn.Sequential(
            # TODO check if it helps, and WTF is ceil_mode.
            # nn.AvgPool2d(kernel_size=pool_size, stride=pool_stride, ceil_mode=True),
            nn.Conv2d(in_channels=self.n_patches, out_channels=n_channels, kernel_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=intermediate_n_features, out_features=N_CLASSES)
        )

    def forward(self, x):
        embedding = self.patch_based_embedding(x)
        scores = self.classifier(embedding)
        return scores


class PatchBasedEmbedding(nn.Module):
    """
    Calculating the k-nearest-neighbors is implemented as convolution with bias, as was done in
    The Unreasonable Effectiveness of Patches in Deep Convolutional Kernels Methods
    (https://arxiv.org/pdf/2101.07528.pdf)
    Details can be found in the Appendix B (page 13).
    """

    def __init__(self,
                 kernel_convolution: torch.Tensor,
                 bias_convolution: torch.Tensor,
                 pool_size: int,
                 pool_stride: int,
                 k_neighbors: int):
        super(PatchBasedEmbedding, self).__init__()
        self.kernel_convolution = nn.Parameter(kernel_convolution, requires_grad=False)
        self.bias_convolution = nn.Parameter(bias_convolution, requires_grad=False)
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.k_neighbors = k_neighbors

    def forward(self, images):
        squared_distances = F.conv2d(images, -1 * self.kernel_convolution) + self.bias_convolution
        embedding = calculate_smaller_than_kth_value_mask(squared_distances, self.k_neighbors).float()
        return embedding


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


def train_model(model, dataloaders, num_epochs, device, criterion, optimizer, log_interval):
    best_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    total_time = 0
    training_step = 0
    interval_accumulator = Accumulator()
    epoch_accumulator = Accumulator()

    for epoch in range(num_epochs):
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
                logger.info(f'{training_step=}'
                            f'loss={interval_accumulator.get_mean_loss():.4f} '
                            f'acc={interval_accumulator.get_accuracy():.2f}%')
                interval_accumulator.reset()

        epoch_test_loss, epoch_test_accuracy = evaluate_model(model, criterion, dataloaders['test'], device)
        wandb.log(data={'test_accuracy': epoch_test_accuracy, 'test_loss': epoch_test_loss}, step=training_step)

        # if the current model reached the best results so far, deep copy the weights of the model.
        if epoch_test_accuracy > best_accuracy:
            best_accuracy = epoch_test_accuracy
            best_weights = copy.deepcopy(model.state_dict())

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


def validate_args(args):
    pass


def get_conv_kernel_and_bias(batch_size, n_patches, patch_size):
    clean_dataloaders = get_dataloaders(batch_size,
                                        normalize_to_unit_gaussian=False,
                                        normalize_to_plus_minus_one=True,
                                        random_crop=False,
                                        random_horizontal_flip=False,
                                        random_erasing=False,
                                        random_resized_crop=False)

    patches = sample_random_patches(clean_dataloaders['train'],
                                    n_patches=n_patches,
                                    patch_size=patch_size,
                                    visualize=True)

    patches_flattened = patches.reshape(patches.shape[0], -1)
    patches_norms_squared = np.linalg.norm(patches_flattened, axis=1) ** 2
    bias_convolution = patches_norms_squared.reshape(1, -1, 1, 1)

    patches = torch.from_numpy(patches)
    bias_convolution = torch.from_numpy(bias_convolution)

    return patches, bias_convolution


def main():
    args = parse_args()
    validate_args(args)
    out_dir = create_out_dir(args.path)
    configure_logger(out_dir)

    device = torch.device(args.device)

    # model, model_name = get_model(args)
    # logger.info(f'Starting to train {model_name} '
    #             f'for {args.epochs} epochs '
    #             f'(using {args.device}) | '
    #             f'opt={args.optimizer_type}, '
    #             f'bs={args.batch_size}, '
    #             f'lr={args.learning_rate}, '
    #             f'wd={args.weight_decay}')
    # model = model.to(device)

    kernel, bias = get_conv_kernel_and_bias(args.batch_size, args.n_patches, args.patch_size)
    model = ClassifierOnPatchBasedEmbedding(kernel_convolution=kernel, bias_convolution=bias, pool_size=args.pool_size,
                                            pool_stride=args.pool_stride,
                                            k_neighbors=int(args.k_neighbors_fraction * args.n_patches),
                                            n_channels=args.n_channels)

    wandb.init(project='thesis', config=args)
    wandb.watch(model)

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    augmented_dataloaders = get_dataloaders(args.batch_size,
                                            normalize_to_unit_gaussian=False,
                                            normalize_to_plus_minus_one=True,
                                            random_crop=False,  # TODO change to True
                                            random_horizontal_flip=False,
                                            random_erasing=False,
                                            random_resized_crop=False)

    train_model(model, augmented_dataloaders, args.epochs, device, criterion, optimizer, args.log_interval)


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
    parser.add_argument('--epochs', type=int, default=1500,
                        help=f'Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help=f'Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help=f'Learning-rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help=f'Momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help=f'Weight decay')

    parser.add_argument('--n_patches', type=int, default=10,
                        help=f'The number of patches')
    parser.add_argument('--patch_size', type=int, default=6,
                        help=f'The size of the patches')
    parser.add_argument('--pool_size', type=int, default=5,
                        help=f'The size of the average-pooling layer after the patch-based-embedding')
    parser.add_argument('--pool_stride', type=int, default=3,
                        help=f'The stride of the average-pooling layer after the patch-based-embedding')
    parser.add_argument('--k_neighbors_fraction', type=float, default=0.4,
                        help=f'which k to use in the k-nearest-neighbors, as a fraction of the total number of patches')

    # Arguments for logging the training process.
    parser.add_argument('--path', type=str, default='./experiments',
                        help=f'Output path for the experiment - '
                             f'a sub-directory named with the data and time will be created within')
    parser.add_argument('--log_interval', type=int, default=3,
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
