import copy
import math
import sys
import wandb
import torch
import faiss

import numpy as np
import pandas as pd
import plotly.express as px
import torchmetrics as tm
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger
from typing import Optional, List, Union, Tuple, Dict, Type
from pathlib import Path
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms import ToTensor, RandomCrop, RandomHorizontalFlip, Normalize, Compose
from pytorch_lightning import LightningDataModule, Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

from consts import N_CLASSES
from patches import sample_random_patches
from schemas.architecture import ArchitectureArgs
from schemas.data import DataArgs
from schemas.dmh import Args, DMHArgs
from schemas.optimization import OptimizationArgs
from utils import (configure_logger, get_args, get_model_device, power_minus_1, get_mlp, get_dataloaders,
                   whiten_data, normalize_data)
from vgg import get_vgg_model_kernel_size, get_vgg_blocks, configs


class KNearestPatchesEmbedding(nn.Module):
    def __init__(self, kernel: np.ndarray, bias: np.ndarray, stride: int, padding: int, k: int,
                 return_as_mask: bool = True):
        """Calculating the k-nearest-neighbors for patches in the input image.

        Calculating the k-nearest-neighbors is implemented as a convolution layer, as in
        The Unreasonable Effectiveness of Patches in Deep Convolutional Kernels Methods
        (https://arxiv.org/pdf/2101.07528.pdf)
        Details can be found in Appendix B (page 13).

        Args:
            kernel: The kernel that will be used during the embedding calculation.
                For example, when using the embedding on the original patches (not-whitened) the kernel will
                be the patches themselves in the patches-dictionary. if we use whitening then the kernel is
                the patches multiplied by WW^T and the bias is the squared-norm of patches multiplied by W (no W^T).
            bias: The bias that will be used during the embedding calculation.
                For example, when using the embedding on the original patches (not-whitened) the bias will
                be the squared-norms of the patches in the patches-dictionary.
        """
        super(KNearestPatchesEmbedding, self).__init__()

        self.k: int = k
        self.stride: int = stride
        self.padding: int = padding
        self.return_as_mask: bool = return_as_mask

        self.kernel = nn.Parameter(torch.Tensor(kernel), requires_grad=False)
        self.bias = nn.Parameter(torch.Tensor(bias), requires_grad=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass.

        Args:
            images: The input tensor of shape (B, C, H, W) to calculate the k-nearest-neighbors.

        Returns:
            A tensor of shape (B, N, H, W) where N is the size of the patches-dictionary.
            The ij spatial location will hold the mask indicating the k nearest neighbors of the patch centered at ij.
        """
        # In every spatial location ij, we'll have a vector containing the squared distances to all the patches.
        # Note that it's not really the squared distance, but the squared distance minus the squared-norm of the
        # input patch in that location, but minimizing this value will minimize the distance
        # (since the norm of the input patch is the same among all patches in the bank).
        distances = F.conv2d(images, self.kernel, self.bias, self.stride, self.padding)
        if self.return_as_mask:
            values = distances.kthvalue(k=self.k, dim=1, keepdim=True).values
            mask = torch.le(distances, values).float()
            return mask
        else:
            indices = torch.topk(distances, k=self.k, dim=1, largest=False).indices
            return indices


class NeighborsValuesAssigner(nn.Module):
    def __init__(self, centroids: np.ndarray, values: np.ndarray, stride: int, padding: int, k: int,
                 use_faiss: bool = False, no_reduction: bool = False, use_linear_function: str = 'none'):
        super(NeighborsValuesAssigner, self).__init__()
        self.kernel_size = centroids.shape[-1]
        self.stride = stride
        self.padding = padding
        self.k = k
        self.use_faiss = use_faiss
        self.no_reduction = no_reduction
        self.use_linear_function = use_linear_function

        if self.use_faiss:
            centroids_flat = centroids.reshape(centroids.shape[0], -1)
            self.index = faiss.IndexFlatL2(centroids_flat.shape[1])
            self.index.add(centroids_flat)
        else:
            kernel = -1 * centroids
            bias = 0.5 * (np.linalg.norm(centroids.reshape(centroids.shape[0], -1), axis=1) ** 2)
            self.knn_indices_calculator = KNearestPatchesEmbedding(kernel, bias, stride, padding, k,
                                                                   return_as_mask=False)

        self.values = nn.Parameter(torch.Tensor(values), requires_grad=False)
        if use_linear_function == 'full':
            values_dim = self.values.shape[1]
            self.conv = nn.Conv2d(in_channels=k*values_dim, out_channels=values_dim, kernel_size=(1, 1))
        elif use_linear_function == 'partial':
            self.conv = nn.Conv2d(in_channels=k, out_channels=1, kernel_size=(1, 1))
        else:  # use_linear_function == 'none'
            self.conv = None

    def forward_using_torch(self, x: torch.Tensor) -> torch.Tensor:
        values_dim = self.values.shape[1]
        knn_indices = self.knn_indices_calculator(x)
        batch_size, k, output_height, output_width = knn_indices.shape
        assert k == self.k, f'self.knn_indices_calculator did not return {self.k} indices'

        result = self.values[knn_indices]
        result = torch.permute(result, dims=(0, 1, 4, 2, 3))
        if self.no_reduction:
            result = torch.flatten(result, start_dim=1, end_dim=2)
        else:
            if self.use_linear_function == 'full':
                result = torch.flatten(result, start_dim=1, end_dim=2)
                result = self.conv(result)
            elif self.use_linear_function == 'partial':
                result = torch.swapaxes(result, 1, 2)
                result = torch.flatten(result, start_dim=0, end_dim=1)
                result = self.conv(result)
                result = torch.reshape(result, shape=(batch_size, values_dim, output_height, output_width))
            else:
                result = torch.mean(result, dim=1)

        return result

    def forward_using_faiss(self, x: torch.Tensor) -> torch.Tensor:
        x_unfolded = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        batch_size, patch_dim, n_patches = x_unfolded.shape
        values_channels = self.values.shape[1]
        output_spatial_size = int(math.sqrt(n_patches))

        # Transpose from (N, C*H*W, M) to (N, M, C*H*W) and then reshape to (N*M, C*H*W) to have collection of vectors
        # Also make contiguous in memory (required by function kmeans.search).
        x_unfolded = x_unfolded.transpose(dim0=1, dim1=2).flatten(start_dim=0, end_dim=1).contiguous()
        _, indices = self.index.search(x_unfolded.cpu().numpy(), self.k)
        x_unfolded_outputs = self.values[indices]
        x_unfolded_outputs = x_unfolded_outputs.mean(dim=1)
        x_unfolded_outputs = x_unfolded_outputs.reshape(batch_size, n_patches, values_channels)
        x_unfolded_outputs = x_unfolded_outputs.transpose(dim0=1, dim1=2)

        x_output = x_unfolded_outputs.reshape(batch_size, values_channels, output_spatial_size, output_spatial_size)

        return x_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_faiss:
            return self.forward_using_faiss(x)
        else:
            return self.forward_using_torch(x)

# class LocallyLinearConvImitator(nn.Module):
#     def __init__(self, patches: np.ndarray, out_channels: int, padding: int, k: int):
#         super(LocallyLinearConvImitator, self).__init__()
#
#         self.k = k
#         self.out_channels = out_channels
#         self.n_patches, self.in_channels, kernel_height, kernel_width = patches.shape
#         self.kernel_size = kernel_height
#         assert kernel_height == kernel_width, "the patches should be square"
#         self.padding = padding
#
#         # TODO DEBUG, since we changed the API to get kernel and bias
#         ipdb.set_trace()
#         self.k_nearest_neighbors_embedding = KNearestPatchesEmbedding(kernel, bias, k, padding=padding)
#
#         # This convolution layer can be thought of as linear function per patch in the patch-dict,
#         # each function is from (in_channels x kernel_size^2) to the reals.
#         self.conv = nn.Conv2d(in_channels=self.in_channels,
#                               out_channels=self.n_patches,  # * self.out_channels,
#                               kernel_size=self.kernel_size,
#                               padding=self.padding)
#
#         # This convolution layer can be thought of as transforming each vector containing the output of
#         # the linear classifier corresponding to the nearest-patch in its index and zero elsewhere.
#         # to a vector of size `out_channels` (the objective is the corresponding teacher conv layer).
#         self.conv_1x1 = nn.Conv2d(in_channels=self.n_patches,
#                                   out_channels=self.out_channels,
#                                   kernel_size=(1, 1))
#
#     def forward(self, x):
#         mask = self.k_nearest_neighbors_embedding(x)
#         patches_linear_outputs = self.conv(x)
#         kth_nearest_patch_linear_output = mask * patches_linear_outputs
#         output = self.conv_1x1(kth_nearest_patch_linear_output)
#         return output
#
#
# def is_conv_block(block: nn.Module):
#     return isinstance(block, nn.Sequential) and isinstance(block[0], nn.Conv2d)
#
#
# class LocallyLinearImitatorVGG(pl.LightningModule):
#
#     def __init__(self, teacher: "LitVGG", datamodule: pl.LightningDataModule, args: Args):
#         super(LocallyLinearImitatorVGG, self).__init__()
#         self.args: Args = args
#         self.save_hyperparameters(args.flattened_dict())
#
#         self.teacher: LitVGG = teacher.requires_grad_(False)
#
#         losses: List[Union[nn.MSELoss, None]] = list()
#         layers: List[Union[LocallyLinearConvImitator, nn.MaxPool2d]] = list()
#         for i, block in enumerate(teacher.features):
#             if is_conv_block(block):
#                 conv_layer: nn.Conv2d = block[0]
#                 patches = sample_random_patches(datamodule.train_dataloader(),
#                                                 args.dmh.n_patches,
#                                                 conv_layer.kernel_size[0],
#                                                 teacher.get_sub_model(i),
#                                                 random_patches=args.dmh.random_patches)
#                 patches_flat = patches.reshape(patches.shape[0], -1)
#                 kmeans = faiss.Kmeans(d=patches_flat.shape[1], k=args.dmh.n_clusters, verbose=True)
#                 kmeans.train(patches_flat)
#                 centroids = kmeans.centroids.reshape(-1, *patches.shape[1:])
#                 conv_imitator = LocallyLinearConvImitator(centroids,
#                                                           conv_layer.out_channels,
#                                                           conv_layer.padding[0],
#                                                           args.dmh.k)
#                 layers.append(conv_imitator)
#                 losses.append(nn.MSELoss())
#             else:
#                 layers.append(block)
#                 losses.append(None)
#
#         self.features = nn.Sequential(*layers)
#         self.losses = nn.ModuleList(losses)
#
#         self.mlp = copy.deepcopy(teacher.mlp)
#         self.mlp.requires_grad_(False)
#
#         # Apparently the Metrics must be an attribute of the LightningModule, and not inside a dictionary.
#         # This is why we have to set them separately here and then the dictionary will map to the attributes.
#         self.train_accuracy = tm.Accuracy()
#         self.validate_accuracy = tm.Accuracy()
#         self.accuracy = {RunningStage.TRAINING: self.train_accuracy,
#                          RunningStage.VALIDATING: self.validate_accuracy}
#
#         # self.state_dict_copy = copy.deepcopy(self.state_dict())
#
#     def forward(self, x: torch.Tensor):
#         features = self.features(x)
#         logits = self.mlp(features)
#         return logits
#
#     def shared_step(self, batch, stage: RunningStage):
#         x, labels = batch  # Note that we do not use the labels for training, only for logging training accuracy.
#         intermediate_losses = list()
#
#         # Prevent BatchNorm layers from changing `running_mean`, `running_var` and `num_batches_tracked`
#         self.teacher.eval()
#
#         for i, layer in enumerate(self.features):
#             x_output = layer(x)
#             if isinstance(layer, LocallyLinearConvImitator):
#                 x_teacher_output = self.teacher.features[i](x).detach()  # TODO should we really detach here?
#                 curr_loss = self.losses[i](x_output, x_teacher_output)
#                 intermediate_losses.append(curr_loss)
#                 self.log(f'{stage.value}_loss_{i}', curr_loss.item())
#             x = x_output.detach()  # TODO Is it the right way?
#
#         loss = sum(intermediate_losses)
#         self.log(f'{stage.value}_loss', loss / len(intermediate_losses))
#
#         logits = self.mlp(x)
#         accuracy = self.accuracy[stage]
#         accuracy(logits, labels)
#         self.log(f'{stage.value}_accuracy', accuracy)
#
#         # if stage == RunningStage.TRAINING:
#         #     # For debugging purposes - verify that the weights of the model changed.
#         #     new_model_state = copy.deepcopy(self.state_dict())
#         #     for weight_name in new_model_state.keys():
#         #         old_weight = self.state_dict_copy[weight_name]
#         #         new_weight = new_model_state[weight_name]
#         #         if torch.allclose(old_weight, new_weight):
#         #             pass
#         #             # logger.debug(f'Weight \'{weight_name}\' of shape {tuple(new_weight.size())} did not change.')
#         #         else:
#         #             logger.debug(f'Weight \'{weight_name}\' of shape {tuple(new_weight.size())} changed.')
#         #     self.state_dict_copy = copy.deepcopy(new_model_state)
#
#         return loss
#
#     def training_step(self, batch, batch_idx):
#         loss = self.shared_step(batch, RunningStage.TRAINING)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         self.shared_step(batch, RunningStage.VALIDATING)
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.SGD(self.features.parameters(),  # Explicitly optimize only the imitators parameters
#                                     self.args.opt.learning_rate,
#                                     self.args.opt.momentum,
#                                     weight_decay=self.args.opt.weight_decay)
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                                          milestones=self.args.opt.learning_rate_decay_steps,
#                                                          gamma=self.args.opt.learning_rate_decay_gamma)
#         return {'optimizer': optimizer, 'lr_scheduler': scheduler}


class ShufflePixels:
    def __init__(self, keep_rgb_triplets_intact: bool = True):
        """A data transformation which shuffles the pixels of the input image.

        Args:
            keep_rgb_triplets_intact: If it's true, shuffle the RGB triplets and not each value separately.
        """
        self.keep_rgb_triplets_intact = keep_rgb_triplets_intact

    def __call__(self, img):
        assert img.ndim == 3 and img.shape[0] == 3, "The input-image is expected to be of shape 3 x H x W"
        start_dim = 1 if self.keep_rgb_triplets_intact else 0
        img_flat = torch.flatten(img, start_dim=start_dim)
        permutation = torch.randperm(img_flat.shape[-1])
        permuted_img_flat = img_flat[..., permutation]
        permuted_img = torch.reshape(permuted_img_flat, shape=img.shape)
        return permuted_img


class DataModule(LightningDataModule):
    def __init__(self, args: DataArgs, batch_size: int, data_dir: str = "./data"):
        """A datamodule to be used with PyTorch Lightning modules.

        Args:
            args: The data's arguments-schema.
            batch_size: The batch-size.
            data_dir: The data directory to read the data from (if it's not there - downloads it).
        """
        super().__init__()

        self.dataset_class = self.get_dataset_class(args.dataset_name)
        self.n_channels = args.n_channels
        self.spatial_size = args.spatial_size
        self.data_dir = data_dir
        self.batch_size = batch_size

        transforms_list_no_aug, transforms_list_with_aug = DataModule.get_transforms_lists(args)
        self.transforms = {'aug': Compose(transforms_list_with_aug),
                           'no_aug': Compose(transforms_list_no_aug),
                           'clean': ToTensor()}
        self.datasets = {f'{stage}_{aug}': None
                         for stage in ('fit', 'validate')
                         for aug in ('aug', 'no_aug', 'clean')}

    def get_dataset_class(self, dataset_name: str) -> Union[Type[CIFAR10], Type[MNIST], Type[FashionMNIST]]:
        """Gets the class of the dataset, according to the given dataset name.

        Args:
            dataset_name: name of the dataset (CIFAR10, MNIST or FashionMNIST).

        Returns:
            The dataset class.
        """
        if dataset_name == 'CIFAR10':
            return CIFAR10
        elif dataset_name == 'MNIST':
            return MNIST
        elif dataset_name == 'FashionMNIST':
            return FashionMNIST
        else:
            raise NotImplementedError(f'Dataset {dataset_name} is not implemented.')

    @staticmethod
    def get_transforms_lists(args: DataArgs) -> Tuple[list, list]:
        """Gets the transformations list to be used in the dataloader.

        Args:
            args: The data's arguments-schema.

        Returns:
            One list is the transformations without augmentation,
            and the other is the transformations with augmentations.
        """
        augmentations = DataModule.get_augmentations_transforms(args.random_horizontal_flip,
                                                                args.random_crop,
                                                                args.spatial_size)
        normalization = DataModule.get_normalization_transform(args.normalization_to_plus_minus_one,
                                                               args.normalization_to_unit_gaussian,
                                                               args.n_channels)
        normalizations_list = list() if (normalization is None) else [normalization]
        crucial_transforms = [ToTensor()]
        post_transforms = [ShufflePixels(args.keep_rgb_triplets_intact)] if args.shuffle_images else list()
        transforms_list_no_aug = crucial_transforms + normalizations_list + post_transforms
        transforms_list_with_aug = augmentations + crucial_transforms + normalizations_list + post_transforms

        return transforms_list_no_aug, transforms_list_with_aug

    @staticmethod
    def get_augmentations_transforms(random_flip: bool, random_crop: bool, spatial_size: int) -> list:
        """Gets the augmentations transformations list to be used in the dataloader.

        Args:
            random_flip: Whether to use random-flip augmentation.
            random_crop: Whether to use random-crop augmentation.
            spatial_size: The spatial-size of the input images (needed for the target-size of the random-crop).

        Returns:
            A list containing the augmentations transformations.
        """
        augmentations_transforms = list()

        if random_flip:
            augmentations_transforms.append(RandomHorizontalFlip())
        if random_crop:
            augmentations_transforms.append(RandomCrop(size=spatial_size, padding=4))

        return augmentations_transforms

    @staticmethod
    def get_normalization_transform(plus_minus_one: bool, unit_gaussian: bool, n_channels: int) -> Optional[Normalize]:
        """Gets the normalization transformation to be used in the dataloader (or None, if no normalization is needed).

        Args:
            plus_minus_one: Whether to normalize the input-images from [0,1] to [-1,+1].
            unit_gaussian: Whether to normalize the input-images to have zero mean and std one (channels-wise).
            n_channels: Number of input-channels for each input-image (3 for CIFAR10, 1 for MNIST/FashionMNIST).

        Returns:
            The normalization transformation (or None, if no normalization is needed).
        """
        assert not (plus_minus_one and unit_gaussian), 'Only one should be given'

        if unit_gaussian:
            if n_channels != 3:
                raise NotImplementedError('Normalization for MNIST / FashionMNIST is not supported. ')
            normalization_values = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
        elif plus_minus_one:
            normalization_values = [(0.5,) * n_channels] * 2  # times 2 because one is mean and one is std
        else:
            return None

        return Normalize(*normalization_values)

    def prepare_data(self):
        """Download the dataset if it's not already in `self.data_dir`.
        """
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
                if self.datasets[k] is None:
                    self.datasets[k] = self.dataset_class(self.data_dir,
                                                          train=(s == 'fit'),
                                                          transform=self.transforms[aug])

    def train_dataloader(self):
        """
        Returns:
             The train dataloader, which is the train-data with augmentations.
        """
        return DataLoader(self.datasets['fit_aug'], batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        """
        Returns:
             The validation dataloader, which is the validation-data without augmentations
             (but possibly has normalization, if the training-dataloader has one).
        """
        return DataLoader(self.datasets['validate_no_aug'], batch_size=self.batch_size, num_workers=4)

    def train_dataloader_no_aug(self):
        """
        Returns:
             The train dataloader without augmentations.
        """
        return DataLoader(self.datasets['fit_no_aug'], batch_size=self.batch_size, num_workers=4, shuffle=True)

    def train_dataloader_clean(self):
        """
        Returns:
             The train dataloader without augmentations and normalizations (i.e. the original images in [0,1]).
        """
        return DataLoader(self.datasets['fit_clean'], batch_size=self.batch_size, num_workers=4, shuffle=True)


class LitVGG(pl.LightningModule):
    def __init__(self, arch_args: ArchitectureArgs, opt_args: OptimizationArgs, data_args: DataArgs):
        """A basic CNN, based on the VGG architecture (and some variants).

        Args:
            arch_args: The arguments for the architecture.
            opt_args: The arguments for the optimization process.
            data_args: The arguments for the input data.
        """
        super(LitVGG, self).__init__()
        layers, n_features = get_vgg_blocks(configs[arch_args.model_name],
                                            data_args.n_channels,
                                            data_args.spatial_size,
                                            arch_args.kernel_size,
                                            arch_args.padding,
                                            arch_args.use_batch_norm,
                                            arch_args.bottle_neck_dimension)
        self.features = nn.Sequential(*layers)
        self.mlp = get_mlp(input_dim=n_features,
                           output_dim=N_CLASSES,
                           n_hidden_layers=arch_args.final_mlp_n_hidden_layers,
                           hidden_dim=arch_args.final_mlp_hidden_dim,
                           use_batch_norm=arch_args.use_batch_norm,
                           organize_as_blocks=True)
        self.loss = torch.nn.CrossEntropyLoss()

        self.arch_args: ArchitectureArgs = arch_args
        self.opt_args: OptimizationArgs = opt_args

        self.save_hyperparameters(arch_args.dict())
        self.save_hyperparameters(opt_args.dict())
        self.save_hyperparameters(data_args.dict())

        self.num_blocks = len(self.features) + len(self.mlp)

        self.train_accuracy = tm.Accuracy()
        self.validate_accuracy = tm.Accuracy()
        self.accuracy = {RunningStage.TRAINING: self.train_accuracy,
                         RunningStage.VALIDATING: self.validate_accuracy}

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
        logits = self.mlp(features.flatten(start_dim=1))
        return logits

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

        self.accuracy[stage](logits, labels)

        self.log(f'{stage.value}_loss', loss)
        self.log(f'{stage.value}_accuracy', self.accuracy[stage])

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

        dataloader = get_dataloaders(batch_size=8)["train"]
        x, _ = next(iter(dataloader))
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
        self.output_dim = N_CLASSES
        self.n_hidden_layers = arch_args.final_mlp_n_hidden_layers
        self.hidden_dim = arch_args.final_mlp_hidden_dim
        self.mlp = get_mlp(self.input_dim, self.output_dim, self.n_hidden_layers, self.hidden_dim,
                           use_batch_norm=True, organize_as_blocks=True)
        self.loss = torch.nn.CrossEntropyLoss()

        self.arch_args = arch_args
        self.opt_args = opt_args
        self.save_hyperparameters(arch_args.dict())
        self.save_hyperparameters(opt_args.dict())
        self.save_hyperparameters(data_args.dict())

        self.num_blocks = len(self.mlp)

        # Apparently the Metrics must be an attribute of the LightningModule, and not inside a dictionary.
        # This is why we have to set them separately here and then the dictionary will map to the attributes.
        self.train_accuracy = tm.Accuracy()
        self.validate_accuracy = tm.Accuracy()
        self.accuracy = {RunningStage.TRAINING: self.train_accuracy,
                         RunningStage.VALIDATING: self.validate_accuracy}

    def forward(self, x: torch.Tensor):
        return self.mlp(x)

    def shared_step(self, batch, stage: RunningStage):
        inputs, labels = batch
        logits = self(inputs)
        loss = self.loss(logits, labels)

        self.accuracy[stage](logits, labels)

        self.log(f'{stage.value}_loss', loss)
        self.log(f'{stage.value}_accuracy', self.accuracy[stage])

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


def get_estimates_matrix(data: torch.Tensor, k: int):
    """Calculates a matrix containing the intrinsic-dimension estimators.

    The ij-th cell contains the j-th estimate for the i-th data-point.
    In the notation of the paper below it's $\\hat(m)_j(x_i)$.

    See `Maximum Likelihood Estimation of Intrinsic Dimension
    <https://papers.nips.cc/paper/2004/file/74934548253bcab8490ebd74afed7031-Paper.pdf>`_
    """
    assert data.ndim == 2, f"data has shape {tuple(data.shape)}, expected (n, d) i.e. n d-dimensional vectors. "

    if k > data.shape[0]:
        logger.debug(f"Number of data-points is {data.shape[0]} and k={k} should be smaller. ")
        k = data.shape[0] - 1
        logger.debug(f"k was changed to {k}")

    distance_matrix = torch.cdist(data, data)

    distances, _ = torch.topk(distance_matrix, k=1 + k, largest=False)
    distances = distances[:, 1:]  # Remove the 1st column corresponding to the (zero) distance between item and itself.
    log_distances = torch.log(distances)
    log_distances_cumsum = torch.cumsum(log_distances, dim=1)
    log_distances_cummean = torch.divide(log_distances_cumsum,
                                         torch.arange(start=1, end=log_distances.shape[1] + 1,
                                                      device=log_distances.device))
    log_distances_cummean_shifted = F.pad(log_distances_cummean[:, :-1], (1, 0))
    log_distances_minus_means = log_distances - log_distances_cummean_shifted
    estimates = power_minus_1(log_distances_minus_means)

    return estimates


def calc_intrinsic_dimension(data: torch.Tensor, k1: int, k2: int) -> float:
    """Calculates the intrinsic-dimension of the data, which is the mean k-th estimators from k1 to k2.

    See the end of section 3 in `Maximum Likelihood Estimation of Intrinsic Dimension
    <https://papers.nips.cc/paper/2004/file/74934548253bcab8490ebd74afed7031-Paper.pdf>`_
    """
    estimates = get_estimates_matrix(data, k2)
    estimate_mean_over_data_points = torch.mean(estimates, dim=0)
    estimate_mean_over_k1_to_k2 = torch.mean(estimate_mean_over_data_points[k1:k2 + 1])

    return estimate_mean_over_k1_to_k2.item()


def indices_to_mask(n, indices, negate=False):
    # TODO Report PyTorch BUG when indices is empty :(
    # mask = torch.scatter(torch.zeros(n, dtype=torch.bool), dim=0, index=indices, value=1)
    mask = torch.zeros(n, dtype=torch.bool, device=indices.device).scatter_(dim=0, index=indices, value=1)
    if negate:
        mask = torch.bitwise_not(mask)
    return mask


def get_flattened_patches(dataloader, n_patches, kernel_size,
                          shuffle_before_estimate: bool = False, sub_model=None, device=None):
    patches = sample_random_patches(dataloader, n_patches, kernel_size, sub_model)
    patches = patches.reshape(patches.shape[0], -1)  # a.k.a. flatten in NumPy

    if shuffle_before_estimate:
        patches = np.random.default_rng().permuted(patches, axis=1, out=patches)

    patches = patches.astype(np.float64)  # Increase accuracy of calculations.
    patches = torch.from_numpy(patches)

    if device is not None:
        patches = patches.to(device)

    return patches


def get_patches_to_keep_mask(patches, minimal_distance: float = 1e-05):
    distance_matrix = torch.cdist(patches, patches)
    small_distances_indices = torch.nonzero(torch.less(distance_matrix, minimal_distance))
    different_patches_mask = (small_distances_indices[:, 0] != small_distances_indices[:, 1])
    different_patches_close_indices_pairs = small_distances_indices[different_patches_mask]
    different_patches_close_indices = different_patches_close_indices_pairs.unique()
    patches_to_keep_mask = indices_to_mask(len(patches), different_patches_close_indices, negate=True)

    return patches_to_keep_mask


def get_patches_not_too_close_to_one_another(dataloader, n_patches, patch_size,
                                             minimal_distance: float = 1e-5,
                                             shuffle_before_estimate: bool = False,
                                             sub_model=None, 
                                             device=None):
    ratio_to_extend_n = 1.5
    n_patches_extended = math.ceil(n_patches * ratio_to_extend_n)
    patches = get_flattened_patches(dataloader, n_patches_extended, patch_size, 
                                    shuffle_before_estimate, sub_model, device)
    patches_to_keep_mask = get_patches_to_keep_mask(patches, minimal_distance)
    patches = patches[patches_to_keep_mask]

    patches = patches[:n_patches]  # This is done to get exactly (or up-to) n like the user requested

    return patches


def log_dim_per_k_graph(metrics, prefix, estimates):
    min_k = 5
    max_k = len(estimates) + 1
    df = pd.DataFrame(estimates[min_k - 1:], index=np.arange(min_k, max_k), columns=['k-th intrinsic-dimension'])
    df.index.name = 'k'
    metrics[f'{prefix}-int_dim_per_k'] = px.line(df)


def log_singular_values(metrics, prefix, data):
    n, d = data.shape
    data_dict = {
        'original_data': data,
        'normalized_data': normalize_data(data),
        'whitened_data': whiten_data(data),
        'random_data': np.random.default_rng().multivariate_normal(mean=np.zeros(d), cov=np.eye(d), size=n)
    }

    singular_values = dict()
    singular_values_ratio = dict()
    explained_variance_ratio = dict()
    variance_ratio = dict()
    reconstruction_errors = dict()
    for data_name in data_dict.keys():
        data_orig = data_dict[data_name]
        logger.debug(f'Calculating SVD for {data_name} with shape {tuple(data_orig.shape)}...')
        # u is a n x n matrix, the columns are the left singular vectors.
        # s is a vector of size min(n, d), containing the singular values.
        # v is a d x d matrix, the *rows* are the right singular vectors.
        u, s, v_t = np.linalg.svd(data_orig)
        v = v_t.T  # d x d matrix, now the *columns* are the right singular vectors.
        logger.debug('Finished calculating SVD')

        singular_values[data_name] = s

        # Inspiration taken from sklearn.decomposition._pca.PCA._fit_full
        explained_variance = (s ** 2) / (n - 1)
        explained_variance_ratio[data_name] = explained_variance / explained_variance.sum()

        # Inspiration taken from "The Unreasonable Effectiveness of Patches in Deep Convolutional Kernels Methods"
        variance_ratio[data_name] = np.cumsum(s) / np.sum(s)
        singular_values_ratio[data_name] = s / s[0]

        logger.debug('Calculating reconstruction error...')
        transformed_data = np.dot(data_orig, v)  # n x d matrix (like the original)
        reconstruction_errors_list = list()
        for k in range(1, 101):
            v_reduced = v[:, :k]  # d x k matrix
            transformed_data_reduced = np.dot(transformed_data, v_reduced)  # n x k matrix
            transformed_data_reconstructed = np.dot(transformed_data_reduced, v_reduced.T)  # n x d matrix
            data_reconstructed = np.dot(transformed_data_reconstructed, v_t)  # n x d matrix
            reconstruction_error = np.linalg.norm(data_orig - data_reconstructed)
            reconstruction_errors_list.append(reconstruction_error)
        logger.debug('Finished calculating reconstruction error.')

        reconstruction_errors[data_name] = np.array(reconstruction_errors_list)

    # Inspiration (and the name 'd_cov') taken from
    # "The Unreasonable Effectiveness of Patches in Deep Convolutional Kernels Methods"
    metrics[f'{prefix}-d_cov'] = np.where(variance_ratio['original_data'] > 0.95)[0][0]

    fig_args = dict(markers=True)
    metrics[f'{prefix}-singular_values'] = px.line(pd.DataFrame(singular_values), **fig_args)
    metrics[f'{prefix}-singular_values_ratio'] = px.line(pd.DataFrame(singular_values_ratio), **fig_args)
    metrics[f'{prefix}-variance_ratio'] = px.line(pd.DataFrame(variance_ratio), **fig_args)
    metrics[f'{prefix}-explained_variance_ratio'] = px.line(pd.DataFrame(explained_variance_ratio), **fig_args)
    metrics[f'{prefix}-reconstruction_errors'] = px.line(pd.DataFrame(reconstruction_errors), **fig_args)


def log_final_estimate(metrics, estimates, extrinsic_dimension, block_name, k1, k2):
    estimate_mean_over_k1_to_k2 = torch.mean(estimates[k1:k2 + 1])
    intrinsic_dimension = estimate_mean_over_k1_to_k2.item()
    dimensions_ratio = intrinsic_dimension / extrinsic_dimension

    block_name = f'{block_name}-ext_dim_{extrinsic_dimension}'
    metrics.update({f'{block_name}-int_dim': intrinsic_dimension,
                    f'{block_name}-dim_ratio': dimensions_ratio})
    return intrinsic_dimension, dimensions_ratio


class IntrinsicDimensionCalculator(Callback):

    def __init__(self, args: DMHArgs, minimal_distance: float = 1e-05):
        """
        Since the intrinsic-dimension calculation takes logarithm of the distances,
        if they are zero (or very small) it can cause numerical issues (NaN).
        """
        self.args: DMHArgs = args
        self.minimal_distance = minimal_distance

        # Since the intrinsic-dimension calculation takes logarithm of the distances,
        # if they are zero (or very small) it can cause numerical issues (NaN).
        # The solution is to sample a bit more patches than requested,
        # and later we remove patches that are really close to one another,
        # and we want our final number of patches to be the desired one.
        ratio_to_extend_n = 1.5 if self.args.estimate_dim_on_patches else 1.01  # Lower probability to get similar images.
        self.n_clusters_extended = math.ceil(self.args.n_clusters * ratio_to_extend_n)

    def calc_int_dim_per_layer_on_dataloader(self, trainer, pl_module, dataloader, log_graphs: bool = False):
        """
        Given a VGG model, go over each block in it and calculates the intrinsic dimension of its input data.
        """

        log_graphs = log_graphs or (trainer.global_step == 0)

        metrics = dict()
        for i in range(pl_module.num_blocks):
            block_name = f'block_{i}'
            if self.args.estimate_dim_on_images or (i >= len(pl_module.kernel_sizes)):
                patch_size = -1
            else:
                patch_size = pl_module.kernel_sizes[i]
            patches = get_patches_not_too_close_to_one_another(dataloader, self.args.n_clusters, patch_size, 
                                                               self.minimal_distance,
                                                               self.args.shuffle_before_estimate,
                                                               pl_module.get_sub_model(i))
            
            estimates_matrix = get_estimates_matrix(patches, 8*self.args.k2 if log_graphs else self.args.k2)
            estimates = torch.mean(estimates_matrix, dim=0)

            if log_graphs:
                log_dim_per_k_graph(metrics, block_name, estimates.cpu().numpy())
                log_singular_values(metrics, block_name, patches.cpu().numpy())

            mle_int_dim, ratio = log_final_estimate(metrics, estimates, patches.shape[1], block_name, self.args.k1, self.args.k2)
            logger.debug(f'epoch {trainer.current_epoch:0>2d} block {i:0>2d} '
                         f'mle_int_dim {mle_int_dim:.2f} ({100 * ratio:.2f}% of ext_sim {patches.shape[1]})')
        trainer.logger.experiment.log(metrics, step=trainer.global_step, commit=False)

    def calc_int_dim_per_layer(self, trainer, pl_module, log_graphs: bool = False):
        """
        Calculate the intrinsic-dimension of across layers on the validation-data (without augmentations).
        When the callback is called from the fit loop (e.g. on_fit_begin / on_fit_end) it's in training-mode,
        so the dropout / batch-norm layers are still training. When it's being called from the validation-loop (e.g.
        in on_validation_epoch_end) the training-mode is off.
        We change the training-mode explicitly to False, and set it back like it was before after we finish.
        """
        training_mode = pl_module.training
        pl_module.eval()
        self.calc_int_dim_per_layer_on_dataloader(trainer, pl_module,
                                                  dataloader=trainer.request_dataloader(RunningStage.VALIDATING)[1],
                                                  log_graphs=log_graphs)
        pl_module.train(training_mode)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_step == 0:  # This happens in the end of validation loop sanity-check before training,
            return  # and we do not want to treat it the same as actual validation-epoch end.
        self.calc_int_dim_per_layer(trainer, pl_module)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Since log_graphs=True takes a lot of time, we do it only in the beginning /end of the training process.
        """
        self.calc_int_dim_per_layer(trainer, pl_module, log_graphs=True)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Since log_graphs=True takes a lot of time, we do it only in the beginning /end of the training process.
        """
        self.calc_int_dim_per_layer(trainer, pl_module, log_graphs=True)


class LinearRegionsCalculator(Callback):

    def __init__(self, args: DMHArgs):
        """
        Since the intrinsic-dimension calculation takes logarithm of the distances,
        if they are zero (or very small) it can cause numerical issues (NaN).
        """
        self.args: DMHArgs = args

    @torch.no_grad()
    def calc_activations(self, patches: np.ndarray, block: nn.Sequential):
        patches_tensor = torch.from_numpy(patches).to(get_model_device(block))
        patches_activations = block(patches_tensor).squeeze(dim=-1).squeeze(dim=-1)
        return patches_activations

    @torch.no_grad()
    def log_linear_regions_respect(self, metrics, pl_module: LitVGG, dataloader, block_index):
        """
        TODO Might be interesting to perform kmeans to decreasing number of centroids 
             and plot knee curve and linear-regions respectfullness.
        
        TODO Can also iterate the whole dataset's patches and count the actual good/bad patches
        """
        block = pl_module.features[block_index]
        patches = sample_random_patches(dataloader, self.args.n_patches, 
                                        pl_module.kernel_sizes[block_index], 
                                        pl_module.get_sub_model(block_index),
                                        verbose=False)
        patches_flat = patches.reshape(patches.shape[0], -1)

        kmeans = faiss.Kmeans(d=patches_flat.shape[1], k=self.args.n_clusters)
        kmeans.train(patches_flat)
        centroids = kmeans.centroids.reshape(-1, *patches.shape[1:])
        _, indices = kmeans.assign(patches_flat)
        random_indices = np.random.default_rng().choice(self.args.n_clusters, size=self.args.n_patches)
        
        patches_activations = self.calc_activations(patches, block)
        centroids_activations = self.calc_activations(centroids, block)
        centroids_of_patches_activations = centroids_activations[indices]
        random_centroids_activations = centroids_activations[random_indices]

        patches_active_neurons = (patches_activations > 0)
        centroids_active_neurons = (centroids_of_patches_activations > 0)
        random_centroids_active_neurons = (random_centroids_activations > 0)
        different_activations = (patches_active_neurons != centroids_active_neurons)
        different_activations_to_random_centroids = (patches_active_neurons != random_centroids_active_neurons)
        fraction_different_activations = torch.mean(different_activations.float(), dim=1)
        fraction_different_activations_random = torch.mean(different_activations_to_random_centroids.float(), dim=1)
        
        # This is the number of patches which are in a different linear region than their matched centroid
        n_bad_patches = torch.count_nonzero(fraction_different_activations)
        n_good_patches = fraction_different_activations.numel() - n_bad_patches
        n_bad_patches_random = torch.count_nonzero(fraction_different_activations_random)
        n_good_patches_random = fraction_different_activations_random.numel() - n_bad_patches_random
        
        fraction_good_patches = n_good_patches / self.args.n_patches
        sym_diff_mean = torch.mean(fraction_different_activations)
        sym_diff_median = torch.median(fraction_different_activations)
        fraction_good_patches_random = n_good_patches_random / self.args.n_patches
        sym_diff_mean_random = torch.mean(fraction_different_activations_random)
        sym_diff_median_random = torch.median(fraction_different_activations_random)
        
        metrics[f'block_{block_index}_fraction_good_patches'] = fraction_good_patches.item()
        metrics[f'block_{block_index}_sym_diff_mean'] = sym_diff_mean.item()
        metrics[f'block_{block_index}_sym_diff_median'] = sym_diff_median.item()

        metrics[f'block_{block_index}_fraction_good_patches_random_centroids'] = fraction_good_patches_random.item()
        metrics[f'block_{block_index}_sym_diff_mean_random_centroids'] = sym_diff_mean_random.item()
        metrics[f'block_{block_index}_sym_diff_median_random_centroids'] = sym_diff_median_random.item()

    def log_linear_regions_respect_per_layer(self, trainer, pl_module: LitVGG, dataloader):
        """
        Given a VGG model, go over each block in it and calculates the intrinsic dimension of its input data.
        """
        metrics = dict()
        for i in range(len(pl_module.features)):
            self.log_linear_regions_respect(metrics, pl_module, dataloader, block_index=i)
        trainer.logger.experiment.log(metrics, step=trainer.global_step, commit=True)  # TODO maybe pl_module.log()
        # pl_module.log_dict(metrics)  # TODO maybe pl_module.log()

    def on_validation_epoch_end(self, trainer, pl_module: LitVGG):
        self.log_linear_regions_respect_per_layer(trainer, pl_module,
                                                  dataloader=trainer.request_dataloader(RunningStage.VALIDATING)[1])


def initialize_model(args: Args, wandb_logger: WandbLogger):
    if args.arch.model_name.startswith('VGG'):
        model_class = LitVGG
    else:
        model_class = LitMLP

    if args.arch.use_pretrained:
        artifact = wandb_logger.experiment.use_artifact(args.arch.pretrained_path, type='model')
        artifact_dir = artifact.download()
        model = model_class.load_from_checkpoint(str(Path(artifact_dir) / "model.ckpt"),
                                                 arch_args=args.arch, opt_args=args.opt, data_args=args.data)
    else:
        model = model_class(args.arch, args.opt, args.data)

    return model


def initialize_wandb_logger(args: Args, name_suffix: str = ''):
    return WandbLogger(project='thesis',
                       config=args.flattened_dict(),
                       name=args.env.wandb_run_name + name_suffix,
                       log_model=True)


def initialize_trainer(args: Args, wandb_logger: WandbLogger):
    checkpoint_callback = ModelCheckpoint(monitor='validate_accuracy', mode='max')
    model_summary_callback = ModelSummary(max_depth=3)
    callbacks = [checkpoint_callback, model_summary_callback]
    if isinstance(args.env.multi_gpu, list) or (args.env.multi_gpu != 0):
        trainer_kwargs = dict(gpus=args.env.multi_gpu, strategy="dp")
    else:
        trainer_kwargs = dict(gpus=[args.env.device_num]) if args.env.is_cuda else dict()
    if args.env.debug:
        trainer_kwargs.update({f'limit_{t}_batches': 3 for t in ['train', 'val']})
        trainer_kwargs.update({'log_every_n_steps': 1})
    if args.dmh.estimate_intrinsic_dimension:
        callbacks.append(IntrinsicDimensionCalculator(args.dmh))
    if args.dmh.linear_regions_calculator:
        callbacks.append(LinearRegionsCalculator(args.dmh))
    trainer = pl.Trainer(logger=wandb_logger, callbacks=callbacks, max_epochs=args.opt.epochs,
                         **trainer_kwargs)
    return trainer


class ImitatorKNN(pl.LightningModule):
    def __init__(self, teacher: LitVGG, args: Args, datamodule: DataModule):
        super().__init__()

        self.args: Args = args
        self.datamodule = datamodule

        self.imitators: nn.Sequential = nn.Sequential()
        self.features: nn.Sequential = copy.deepcopy(teacher.features)
        self.flatten: nn.Flatten = nn.Flatten()
        self.mlp: nn.Sequential = copy.deepcopy(teacher.mlp)

        self.imitated_blocks_output_channels: List[int] = list()
        self.imitated_blocks_conv_kwargs: List[Dict[str, int]] = list()

        self.loss = nn.CrossEntropyLoss()

    @torch.no_grad()
    def imitate_first_block(self):
        assert len(self.features) > 0, "This function should be called when there is some block to imitate."
        teacher_block = self.features[0]
        self.features = self.features[1:]

        teacher_block.requires_grad_(False)
        teacher_block.eval()

        assert isinstance(teacher_block, nn.Sequential), 'No support for MaxPool/AvgPool as a separate block.'
        teacher_conv = teacher_block[0]
        assert isinstance(teacher_conv, nn.Conv2d), 'First layer in the block must be convolution-layer.'

        self.imitated_blocks_output_channels.append(teacher_conv.out_channels)
        self.imitated_blocks_conv_kwargs.append({k: getattr(teacher_conv, k)
                                                for k in ['kernel_size', 'dilation', 'padding', 'stride']})

        # TODO Consider doing the following:
        #  Whitening / normalizing the patches.
        if self.args.dmh.dataset_type_for_patches_dictionary == 'aug':
            dataloader = self.datamodule.train_dataloader()
        elif self.args.dmh.dataset_type_for_patches_dictionary == 'no_aug':
            dataloader = self.datamodule.train_dataloader_no_aug()
        else:  # self.args.dmh.dataset_type_for_patches_dictionary == 'clean'
            dataloader = self.datamodule.train_dataloader_clean()

        patches = sample_random_patches(dataloader, self.args.dmh.n_patches,
                                        teacher_conv.kernel_size[0], self.imitators,
                                        random_uniform_patches=self.args.dmh.random_uniform_patches,
                                        random_gaussian_patches=self.args.dmh.random_gaussian_patches,
                                        verbose=True)
        patches_flat = patches.reshape(patches.shape[0], -1)
        kmeans = faiss.Kmeans(d=patches_flat.shape[1], k=self.args.dmh.n_clusters, verbose=True)
        kmeans.train(patches_flat)
        centroids = kmeans.centroids.reshape(-1, *patches.shape[1:])
        centroids_tensor = torch.from_numpy(centroids).to(self.device)
        centroids_outputs = teacher_block(centroids_tensor)
        centroids_outputs = centroids_outputs.cpu().numpy().squeeze(axis=(2, 3))

        imitator = NeighborsValuesAssigner(
            centroids,
            values=centroids_outputs,
            stride=teacher_conv.stride[0],
            padding=teacher_conv.padding[0],
            k=self.args.dmh.k,
            use_faiss=self.args.dmh.use_faiss,
            no_reduction=self.args.dmh.no_reduction,
            use_linear_function=self.args.dmh.use_linear_function
        )

        loss = 0
        for inputs, _ in tqdm(self.datamodule.val_dataloader(), desc=f'Evaluating imitated block'):
            inputs = self.imitators(inputs)
            batch_targets = teacher_block(inputs)
            batch_imitated_targets = imitator(inputs)
            loss += F.l1_loss(batch_targets, batch_imitated_targets)
        loss /= len(self.datamodule.val_dataloader())

        self.imitators = nn.Sequential(*self.imitators, imitator)

        return loss

    def forward(self, x: torch.Tensor):
        x = self.imitators(x)
        x = self.features(x)
        x = self.flatten(x)
        logits = self.mlp(x)

        return logits

    def shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: RunningStage):
        """Performs train/validation step, depending on the given `stage`.

        Note that when training in multi-GPU setting, in `DataParallel` strategy, the input `batch` will actually
        be only a portion of the input batch.
        We also return the logits and the labels to calculate the accuracy in `shared_step_end`.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            stage: Indicating if this is a training-step or a validation-step.

        Returns:
            A dictionary containing the loss, logits and labels.
        """
        x, labels = batch
        logits = self(x)
        loss = self.loss(logits, labels)
        predictions = torch.argmax(logits, dim=1)
        accuracy = torch.sum(labels == predictions).item() / len(labels)

        self.log(f'{stage.value}_loss', loss)
        self.log(f'{stage.value}_accuracy', accuracy, on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Performs a training-step.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            batch_idx: The index of the batch in the dataset.

        Returns:
            A dictionary containing the loss, logits and labels.
        """
        return self.shared_step(batch, RunningStage.TRAINING)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Performs a validation-step.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            batch_idx: The index of the batch in the dataset.

        Returns:
            A dictionary containing the loss, logits and labels.
        """
        return self.shared_step(batch, RunningStage.VALIDATING)

    def configure_optimizers(self):
        """Configure the optimizer and the learning-rate scheduler for the training process.

        Returns:
            A dictionary containing the optimizer and learning-rate scheduler.
        """
        optimizer = torch.optim.SGD(self.parameters(),
                                    self.args.opt.learning_rate,
                                    self.args.opt.momentum,
                                    weight_decay=self.args.opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.args.opt.learning_rate_decay_steps,
                                                         gamma=self.args.opt.learning_rate_decay_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


def initialize_datamodule(args: DataArgs, batch_size: int):
    datamodule = DataModule(args, batch_size)
    datamodule.prepare_data()
    datamodule.setup(stage='fit')
    datamodule.setup(stage='validate')

    return datamodule


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


def main():
    args = get_args(args_class=Args)
    datamodule = initialize_datamodule(args.data, args.opt.batch_size)
    wandb_logger = initialize_wandb_logger(args)
    configure_logger(args.env.path, print_sink=sys.stdout, level='DEBUG')  # if args.env.debug else 'INFO')

    model = initialize_model(args, wandb_logger)
    if not args.arch.use_pretrained:
        wandb_logger.watch(model, log='all')
        trainer = initialize_trainer(args, wandb_logger)
        trainer.fit(model, datamodule=datamodule)
        unwatch_model(model)

    if args.dmh.imitate_with_knn:
        datamodule_for_sampling = initialize_datamodule(args.data, batch_size=4)
        imitator = ImitatorKNN(model, args, datamodule_for_sampling)
        for i in range(len(imitator.features)):
            replaced_block_error = imitator.imitate_first_block()
            wandb_logger = initialize_wandb_logger(args, name_suffix=f'_block_{i}')
            wandb_logger.watch(imitator, log='all')
            wandb_logger.experiment.summary['replaced_block_error'] = replaced_block_error
            trainer = initialize_trainer(args, wandb_logger)
            trainer.fit(imitator, datamodule=datamodule)
            unwatch_model(imitator)

    # if args.dmh.imitate_with_locally_linear_model:
    #     imitator = LocallyLinearImitatorVGG(model, datamodule, args)
    #     wandb_logger.watch(imitator, log='all')
    #     trainer = initialize_trainer(args, wandb_logger)
    #     trainer.fit(imitator, datamodule=datamodule)


if __name__ == '__main__':
    main()
