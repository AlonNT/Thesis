import copy
import math
import sys
import wandb
import torch
import faiss

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import torchmetrics as tm
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger
from typing import Optional, List, Union, Tuple
from pathlib import Path

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms import ToTensor, RandomCrop, RandomHorizontalFlip, Normalize, Compose
from torchvision.transforms.functional import center_crop
from pytorch_lightning import LightningDataModule, Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

from consts import N_CLASSES
from patches import sample_random_patches
from schemas.data import DataArgs
from schemas.dmh import Args, DMHArgs
from utils import (configure_logger, get_args, power_minus_1, get_mlp, get_dataloaders, calc_whitening_from_dataloader,
                   whiten_data, normalize_data, get_covariance_matrix, get_whitening_matrix_from_covariance_matrix,
                   get_args_from_flattened_dict, get_model_output_shape, get_random_initialized_conv_kernel_and_bias)
from vgg import get_vgg_model_kernel_size, get_blocks, configs


class KNearestPatchesEmbedding(nn.Module):
    """
    Calculating the k-nearest-neighbors is implemented as convolution with bias, as was done in
    The Unreasonable Effectiveness of Patches in Deep Convolutional Kernels Methods
    (https://arxiv.org/pdf/2101.07528.pdf)
    Details can be found in Appendix B (page 13).
    """

    def __init__(self,
                 kernel: np.ndarray,
                 bias: np.ndarray,
                 k: int = 1,
                 up_to_k: bool = True,
                 stride: int = 1,
                 padding: int = 0,
                 requires_grad: bool = False,
                 random_embedding: bool = False,
                 kmeans_triangle: bool = False):
        super(KNearestPatchesEmbedding, self).__init__()

        self.k: int = k
        self.up_to_k: bool = up_to_k
        self.stride: int = stride
        self.padding: int = padding
        self.kmeans_triangle: bool = kmeans_triangle

        if random_embedding:
            out_channels, in_channels, kernel_height, kernel_width = kernel.shape
            assert kernel_height == kernel_width, "the kernel should be square"
            kernel, bias = get_random_initialized_conv_kernel_and_bias(in_channels, out_channels, kernel_height)

        self.kernel = nn.Parameter(torch.Tensor(kernel), requires_grad=requires_grad)
        self.bias = nn.Parameter(torch.Tensor(bias), requires_grad=requires_grad)

    def forward(self, images):
        # In every spatial location ij, we'll have a vector containing the squared distances to all the patches.
        # Note that it's not really the squared distance, but the squared distance minus the squared-norm of the
        # input patch in that location, but minimizing this value will minimize the distance
        # (since the norm of the input patch is the same among all patches in the bank).
        distances = F.conv2d(images, self.kernel, self.bias, self.stride, self.padding)
        values, indices = distances.kthvalue(k=self.k, dim=1, keepdim=True)
        if self.kmeans_triangle:
            mask = F.relu(torch.mean(distances, dim=1, keepdim=True) - distances)
        elif self.up_to_k:
            mask = (distances <= values).float()
        else:
            mask = torch.zeros_like(distances).scatter_(dim=1, index=indices, value=1)

        return mask


class LocallyLinearConvImitator(nn.Module):
    def __init__(self, patches: np.ndarray, out_channels: int, padding: int, k: int):
        super(LocallyLinearConvImitator, self).__init__()

        self.k = k
        self.out_channels = out_channels
        self.n_patches, self.in_channels, kernel_height, kernel_width = patches.shape
        self.kernel_size = kernel_height
        assert kernel_height == kernel_width, "the patches should be square"
        self.padding = padding

        # TODO DEBUG, since we changed the API to get kernel and bias
        import ipdb;
        ipdb.set_trace()
        self.k_nearest_neighbors_embedding = KNearestPatchesEmbedding(kernel, bias, k, padding=padding)

        # This convolution layer can be thought of as linear function per patch in the patch-dict,
        # each function is from (in_channels x kernel_size^2) to the reals.
        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.n_patches,  # * self.out_channels,
                              kernel_size=self.kernel_size,
                              padding=self.padding)

        # This convolution layer can be thought of as transforming each vector containing the output of
        # the linear classifier corresponding to the nearest-patch in its index and zero elsewhere.
        # to a vector of size `out_channels` (the objective is the corresponding teacher conv layer).
        self.conv_1x1 = nn.Conv2d(in_channels=self.n_patches,
                                  out_channels=self.out_channels,
                                  kernel_size=(1, 1))

    def forward(self, x):
        mask = self.k_nearest_neighbors_embedding(x)
        patches_linear_outputs = self.conv(x)
        kth_nearest_patch_linear_output = mask * patches_linear_outputs
        output = self.conv_1x1(kth_nearest_patch_linear_output)
        return output


def is_conv_block(block: nn.Module):
    return isinstance(block, nn.Sequential) and isinstance(block[0], nn.Conv2d)


class LocallyLinearImitatorVGG(pl.LightningModule):

    def __init__(self, teacher: "LitVGG", datamodule: pl.LightningDataModule, args: Args):
        super(LocallyLinearImitatorVGG, self).__init__()
        self.args: Args = args
        self.save_hyperparameters(args.flattened_dict())

        self.teacher: LitVGG = teacher.requires_grad_(False)

        losses: List[Union[nn.MSELoss, None]] = list()
        layers: List[Union[LocallyLinearConvImitator, nn.MaxPool2d]] = list()
        for i, block in enumerate(teacher.features):
            if is_conv_block(block):
                conv_layer: nn.Conv2d = block[0]
                patches = sample_random_patches(datamodule.train_dataloader(),
                                                args.dmh.n_patches,
                                                conv_layer.kernel_size[0],
                                                teacher.get_sub_model(i),
                                                random_patches=args.dmh.random_patches)
                patches_flat = patches.reshape(patches.shape[0], -1)
                kmeans = faiss.Kmeans(d=patches_flat.shape[1], k=args.dmh.n_clusters, verbose=True)
                kmeans.train(patches_flat)
                centroids = kmeans.centroids.reshape(-1, *patches.shape[1:])
                conv_imitator = LocallyLinearConvImitator(centroids,
                                                          conv_layer.out_channels,
                                                          conv_layer.padding[0],
                                                          args.dmh.k)
                layers.append(conv_imitator)
                losses.append(nn.MSELoss())
            else:
                layers.append(block)
                losses.append(None)

        self.features = nn.Sequential(*layers)
        self.losses = nn.ModuleList(losses)

        self.mlp = copy.deepcopy(teacher.mlp)
        self.mlp.requires_grad_(False)

        # Apparently the Metrics must be an attribute of the LightningModule, and not inside a dictionary.
        # This is why we have to set them separately here and then the dictionary will map to the attributes.
        self.train_accuracy = tm.Accuracy()
        self.validate_accuracy = tm.Accuracy()
        self.validate_no_aug_accuracy = tm.Accuracy()
        self.accuracy = {RunningStage.TRAINING: [self.train_accuracy],
                         RunningStage.VALIDATING: [self.validate_accuracy, self.validate_no_aug_accuracy]}

        # self.state_dict_copy = copy.deepcopy(self.state_dict())

    def forward(self, x: torch.Tensor):
        features = self.features(x)
        logits = self.mlp(features)
        return logits

    def shared_step(self, batch, stage: RunningStage, dataloader_idx=0):
        x, labels = batch  # Note that we do not use the labels for training, only for logging training accuracy.
        intermediate_losses = list()
        name = stage.value if dataloader_idx == 0 else f'{stage.value}_no_aug'

        # Prevent BatchNorm layers from changing `running_mean`, `running_var` and `num_batches_tracked`
        self.teacher.eval()

        for i, layer in enumerate(self.features):
            x_output = layer(x)
            if isinstance(layer, LocallyLinearConvImitator):
                x_teacher_output = self.teacher.features[i](x).detach()  # TODO should we really detach here?
                curr_loss = self.losses[i](x_output, x_teacher_output)
                intermediate_losses.append(curr_loss)
                self.log(f'{name}_loss_{i}', curr_loss.item())
            x = x_output.detach()  # TODO Is it the right way?

        loss = sum(intermediate_losses)
        self.log(f'{name}_loss', loss / len(intermediate_losses))

        logits = self.mlp(x)
        accuracy = self.accuracy[stage][dataloader_idx]
        accuracy(logits, labels)
        self.log(f'{name}_accuracy', accuracy)

        # if stage == RunningStage.TRAINING:
        #     # For debugging purposes - verify that the weights of the model changed.
        #     new_model_state = copy.deepcopy(self.state_dict())
        #     for weight_name in new_model_state.keys():
        #         old_weight = self.state_dict_copy[weight_name]
        #         new_weight = new_model_state[weight_name]
        #         if torch.allclose(old_weight, new_weight):
        #             pass
        #             # logger.debug(f'Weight \'{weight_name}\' of shape {tuple(new_weight.size())} did not change.')
        #         else:
        #             logger.debug(f'Weight \'{weight_name}\' of shape {tuple(new_weight.size())} changed.')
        #     self.state_dict_copy = copy.deepcopy(new_model_state)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, RunningStage.TRAINING)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        self.shared_step(batch, RunningStage.VALIDATING, dataloader_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.features.parameters(),  # Explicitly optimize only the imitators parameters
                                    self.args.opt.learning_rate,
                                    self.args.opt.momentum,
                                    weight_decay=self.args.opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.args.opt.learning_rate_decay_steps,
                                                         gamma=self.args.opt.learning_rate_decay_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


class LocallyLinearNetwork(pl.LightningModule):

    def __init__(self, args: Args, pre_model: Optional["LocallyLinearNetwork"] = None):
        super(LocallyLinearNetwork, self).__init__()
        self.args: Args = args
        self.save_hyperparameters(args.flattened_dict())
        self.pre_model: Optional["LocallyLinearNetwork"] = pre_model
        self.whitening_matrix: Optional[np.ndarray] = None
        self.patches_covariance_matrix: Optional[np.ndarray] = None
        self.whitened_patches_covariance_matrix: Optional[np.ndarray] = None
        self.input_shape = self.get_input_shape()  # TODO create a dummy image from dataset and pass get_input_shape
        self.input_channels = self.input_shape[0]
        self.input_spatial_size = self.input_shape[1]

        self.embedding = self.init_embedding()
        self.conv = self.init_conv()
        self.avg_pool = self.init_avg_pool()
        self.adaptive_avg_pool = self.init_adaptive_avg_pool()
        self.batch_norm = self.init_batch_norm()
        self.bottle_neck = self.init_bottleneck()
        self.bottle_neck_relu = self.init_bottleneck_relu()
        self.linear = nn.Linear(in_features=self.calc_linear_in_features(), out_features=N_CLASSES)

        self.loss = nn.CrossEntropyLoss()

        # Apparently the Metrics must be an attribute of the LightningModule, and not inside a dictionary.
        # This is why we have to set them separately here and then the dictionary will map to the attributes.
        self.train_accuracy = tm.Accuracy()
        self.validate_accuracy = tm.Accuracy()
        self.validate_no_aug_accuracy = tm.Accuracy()
        self.accuracy = {RunningStage.TRAINING: [self.train_accuracy],
                         RunningStage.VALIDATING: [self.validate_accuracy, self.validate_no_aug_accuracy]}

        self.logits_prediction_mode: bool = True

    def embedding_mode(self):
        self.set_logits_prediction_mode(False)

    def logits_mode(self):
        self.set_logits_prediction_mode(True)

    def set_logits_prediction_mode(self, mode: bool):
        self.logits_prediction_mode = mode

    def get_input_shape(self) -> Tuple[int, int, int]:
        if self.pre_model is None:
            shape = (self.args.data.n_channels,) + (self.args.data.spatial_size,) * 2
        else:
            shape = get_model_output_shape(self.pre_model)

        assert shape[1] == shape[2], 'Should be square'
        return shape

    def init_embedding(self):
        args = self.args.dmh

        if args.replace_embedding_with_regular_conv_relu:
            return nn.Sequential(
                nn.Conv2d(in_channels=self.input_channels,
                          out_channels=args.n_clusters,
                          kernel_size=args.patch_size),
                nn.ReLU()
            )

        kernel, bias = get_random_initialized_conv_kernel_and_bias(in_channels=self.input_channels,
                                                                   out_channels=args.n_clusters,
                                                                   kernel_size=args.patch_size)
        return KNearestPatchesEmbedding(
            kernel, bias, args.k, args.up_to_k,
            stride=args.stride, padding=args.padding,
            requires_grad=args.learnable_embedding,
            random_embedding=args.random_embedding,
            kmeans_triangle=args.kmeans_triangle
        )

    def calculate_embedding_from_data(self, dataloader):
        assert not self.args.dmh.replace_embedding_with_regular_conv_relu, 'This function should not have been called'
        kernel, bias = self.get_kernel_and_bias_from_data(dataloader)
        self.embedding = KNearestPatchesEmbedding(
            kernel, bias, self.args.dmh.k, self.args.dmh.up_to_k,
            stride=self.args.dmh.stride, padding=self.args.dmh.padding,
            requires_grad=self.args.dmh.learnable_embedding,
            random_embedding=self.args.dmh.random_embedding,
            kmeans_triangle=self.args.dmh.kmeans_triangle
        )

    def init_whitening_matrix(self, dataloader: DataLoader) -> Optional[np.ndarray]:
        args = self.args.dmh

        if not args.use_whitening:
            return None

        if args.random_gaussian_patches or args.random_uniform_patches or args.calc_whitening_from_sampled_patches:
            return get_whitening_matrix_from_covariance_matrix(
                self.patches_covariance_matrix, args.whitening_regularization_factor, args.zca_whitening
            )

        return calc_whitening_from_dataloader(dataloader, args.patch_size, args.whitening_regularization_factor,
                                              args.zca_whitening, self.pre_model)

    def init_conv(self) -> Optional[nn.Conv2d]:
        if not self.args.dmh.use_conv:
            return None
        return nn.Conv2d(in_channels=self.input_channels,
                         out_channels=self.args.dmh.n_clusters * self.args.dmh.c,
                         kernel_size=self.args.dmh.patch_size,
                         stride=self.args.dmh.stride,
                         padding=self.args.dmh.padding)

    def init_adaptive_avg_pool(self) -> Optional[nn.AdaptiveAvgPool2d]:
        if not self.args.dmh.use_adaptive_avg_pool:
            return None
        return nn.AdaptiveAvgPool2d(output_size=self.args.dmh.adaptive_pool_output_size)

    def init_avg_pool(self) -> Optional[nn.AvgPool2d]:
        if not self.args.dmh.use_avg_pool:
            return None
        return nn.AvgPool2d(kernel_size=self.args.dmh.pool_size,
                            stride=self.args.dmh.pool_stride,
                            ceil_mode=True)

    def init_batch_norm(self) -> Optional[nn.BatchNorm2d]:
        if not self.args.dmh.use_batch_norm:
            return None
        num_features = self.args.dmh.n_clusters if self.args.dmh.c == 1 else self.args.dmh.c
        return nn.BatchNorm2d(num_features)

    def init_bottleneck(self) -> Optional[nn.Conv2d]:
        if not self.args.dmh.use_bottle_neck:
            return None
        return nn.Conv2d(in_channels=self.args.dmh.n_clusters if (self.args.dmh.c == 1) else self.args.dmh.c,
                         out_channels=self.args.dmh.bottle_neck_dimension,
                         kernel_size=self.args.dmh.bottle_neck_kernel_size)

    def init_bottleneck_relu(self) -> Optional[nn.ReLU]:
        if not self.args.dmh.use_relu_after_bottleneck:
            return None
        return nn.ReLU()

    def calc_linear_in_features(self):
        args = self.args.dmh

        # Inspiration is taken from PyTorch Conv2d docs regarding the output shape
        # https://pytorch.org/docs/1.10.1/generated/torch.nn.Conv2d.html
        embedding_spatial_size = math.floor(
            ((self.input_spatial_size + 2*args.padding - args.patch_size) / args.stride) + 1)

        if args.use_adaptive_avg_pool:
            intermediate_spatial_size = args.adaptive_pool_output_size
        elif args.use_avg_pool:    # ceil and not floor, because we used `ceil_mode=True` in AvgPool2d
            intermediate_spatial_size = math.ceil(1 + (embedding_spatial_size - args.pool_size) / args.pool_stride)
        else:
            intermediate_spatial_size = embedding_spatial_size

        if args.full_embedding:
            patch_dim = self.input_channels * args.patch_size ** 2
            embedding_n_channels = args.n_clusters * patch_dim
        elif args.c > 1:
            embedding_n_channels = args.c
        else:
            embedding_n_channels = args.n_clusters

        intermediate_n_features = embedding_n_channels * (intermediate_spatial_size ** 2)
        bottleneck_output_spatial_size = intermediate_spatial_size - args.bottle_neck_kernel_size + 1
        if args.residual_cat:
            bottle_neck_dimension = args.bottle_neck_dimension + self.input_channels
        else:
            bottle_neck_dimension = args.bottle_neck_dimension
        bottleneck_output_n_features = bottle_neck_dimension * (bottleneck_output_spatial_size ** 2)
        linear_in_features = bottleneck_output_n_features if args.use_bottle_neck else intermediate_n_features
        return linear_in_features

    def get_kernel_and_bias_from_data(self, dataloader: DataLoader):
        # Set the kernel and the bias of the embedding. Note that if we use whitening then the kernel is the patches
        # multiplied by WW^T and the bias is the squared-norm of patches multiplied by W (no W^T).
        kernel = self.get_clustered_patches(dataloader)
        bias = 0.5 * np.linalg.norm(kernel.reshape(kernel.shape[0], -1), axis=1) ** 2
        if self.args.dmh.use_whitening:
            kernel_flat = kernel.reshape(kernel.shape[0], -1)
            kernel_flat = kernel_flat @ self.whitening_matrix.T
            kernel = kernel_flat.reshape(kernel.shape)
        kernel *= -1  # According to the formula as page 13 in https://arxiv.org/pdf/2101.07528.pdf

        return kernel, bias

    def get_full_embedding(self, x: torch.Tensor, features: torch.Tensor, args: DMHArgs):
        x = F.unfold(x, kernel_size=args.patch_size, stride=args.stride, padding=args.padding)
        batch_size, patch_dim, n_patches = x.shape
        spatial_size = int(math.sqrt(n_patches))  # This is the spatial size (e.g. from 28 in CIFAR10 with patch-size 5)
        
        x = x.reshape(batch_size, patch_dim, spatial_size, spatial_size)
        x = torch.repeat_interleave(x, repeats=args.n_clusters, dim=1)
        x = x.reshape(batch_size, patch_dim, args.n_clusters, spatial_size, spatial_size)
        x = x.transpose(1,2)
        x = x.reshape(batch_size, patch_dim*args.n_clusters, spatial_size, spatial_size)

        return x * torch.repeat_interleave(features, repeats=patch_dim, dim=1)

    def forward(self, x: torch.Tensor):
        args = self.args.dmh

        if self.pre_model is not None:
            x = self.pre_model(x)

        features = self.embedding(x)
        if args.full_embedding:
            features = self.get_full_embedding(x, features, args)
        if args.c > 1:
            features = torch.repeat_interleave(features, repeats=args.c, dim=1)
        if args.use_conv:
            features *= self.conv(x)
        if args.c > 1:
            features = features.view(features.shape[0], args.n_clusters, args.c, *features.shape[2:])
            # if args.gather:
            #     import ipdb; ipdb.set_trace()
            #     stop = 'here'
            #     # TODO try to  it reshape from [64, 1024, 8, 28, 28] to [64, 1024, 8, 28, 28]
            #     features_2d = torch.transpose(features, dim0=2, dim1=4).reshape(-1, args.c)
            #     features_nonzero_rows = features_2d[torch.logical_not(torch.all(features_2d == 0, dim=1))].reshape(64,4,28,28,8).transpose(dim0=2, dim1=4)
            #     features = features 
            # else:
            #     features = torch.sum(features, dim=1) / args.k
            features = torch.sum(features, dim=1) / args.k
        if args.use_avg_pool:
            features = self.avg_pool(features)
        if args.use_adaptive_avg_pool:
            features = self.adaptive_avg_pool(features)
        if args.use_batch_norm:
            features = self.batch_norm(features)
        if args.use_bottle_neck:
            features = self.bottle_neck(features)
        if args.use_relu_after_bottleneck:
            features = self.bottle_neck_relu(features)
        if args.residual_add or args.residual_cat:
            if x.shape[-2:] != features.shape[-2:]:  # Spatial size might be slightly different due to lack of padding
                x = center_crop(x, output_size=features.shape[-1])
            features = torch.add(features, x) if args.residual_add else torch.cat((features, x), dim=1)

        if not self.logits_prediction_mode:
            return features

        logits = self.linear(torch.flatten(features, start_dim=1))

        return logits

    def shared_step(self, batch, stage: RunningStage, dataloader_idx=0):
        assert self.logits_prediction_mode, 'Can not do train/validation step when logits prediction mode is off'

        x, labels = batch
        logits = self(x)
        loss = self.loss(logits, labels)

        accuracy = self.accuracy[stage][dataloader_idx]
        accuracy(logits, labels)

        name = stage.value if dataloader_idx == 0 else f'{stage.value}_no_aug'
        self.log(f'{name}_loss', loss)
        self.log(f'{name}_accuracy', accuracy)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, RunningStage.TRAINING)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        self.shared_step(batch, RunningStage.VALIDATING, dataloader_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    self.args.opt.learning_rate,
                                    self.args.opt.momentum,
                                    weight_decay=self.args.opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.args.opt.learning_rate_decay_steps,
                                                         gamma=self.args.opt.learning_rate_decay_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @staticmethod
    def get_diagonal_sum_vs_total_sum_ratio(covariance_matrix):
        non_diagonal_elements_sum = np.sum(np.abs(covariance_matrix - np.diag((np.diagonal(covariance_matrix)))))
        elements_sum = np.sum(np.abs(covariance_matrix))
        ratio = non_diagonal_elements_sum / elements_sum
        return ratio

    def remove_low_norm_patches(self, patches):
        minimal_norm = 0.01
        low_norm_patches_mask = (np.linalg.norm(patches, axis=1) < minimal_norm)
        patches = patches[np.logical_not(low_norm_patches_mask)]
        patches = patches[:self.args.dmh.n_patches]  # Don't leave more than the requested patches

        return patches

    def get_clustered_patches(self, dataloader: DataLoader):
        n_patches_extended = math.ceil(1.01 * self.args.dmh.n_patches)
        patches = sample_random_patches(dataloader, n_patches_extended, self.args.dmh.patch_size,
                                        random_uniform_patches=self.args.dmh.random_uniform_patches,
                                        random_gaussian_patches=self.args.dmh.random_gaussian_patches,
                                        existing_model=self.pre_model)
        patch_shape = patches.shape[1:]
        patches = patches.reshape(patches.shape[0], -1)

        # Low norm (even zero) patches will become problematic later when we normalize by dividing by the norm.
        patches = self.remove_low_norm_patches(patches)

        self.patches_covariance_matrix = get_covariance_matrix(patches)
        self.whitening_matrix = self.init_whitening_matrix(dataloader)

        if self.args.dmh.use_whitening:
            patches = patches @ self.whitening_matrix
            self.whitened_patches_covariance_matrix = get_covariance_matrix(patches)

        if self.args.dmh.n_patches == self.args.dmh.n_clusters:  # This means that we shouldn't do clustering
            if self.args.dmh.normalize_patches_to_unit_vectors:
                patches /= np.linalg.norm(patches, axis=1)[:, np.newaxis]
            return patches.reshape(-1, *patch_shape)
        else:
            kmeans = faiss.Kmeans(d=patches.shape[1], k=self.args.dmh.n_clusters, verbose=True)
            kmeans.train(patches)
            centroids = kmeans.centroids
            if self.args.dmh.normalize_patches_to_unit_vectors:
                centroids /= np.linalg.norm(centroids, axis=1)[:, np.newaxis]
            return centroids.reshape(-1, *patch_shape)

    def unwhiten_patches(self, patches: np.ndarray, only_inv_t: bool = False) -> np.ndarray:
        patches_flat = patches.reshape(patches.shape[0], -1)
        whitening_matrix = np.eye(patches_flat.shape[1]) if (self.whitening_matrix is None) else self.whitening_matrix
        matrix = whitening_matrix.T if only_inv_t else (whitening_matrix @ whitening_matrix.T)
        patches_orig_flat = np.dot(patches_flat, np.linalg.inv(matrix))
        patches_orig = patches_orig_flat.reshape(patches.shape)

        return patches_orig

    @staticmethod
    def get_extreme_patches_indices(norms_numpy, number_of_extreme_patches_to_show):
        partitioned_indices = np.argpartition(norms_numpy, number_of_extreme_patches_to_show)
        worst_patches_indices = partitioned_indices[:number_of_extreme_patches_to_show]
        partitioned_indices = np.argpartition(norms_numpy, len(norms_numpy) - number_of_extreme_patches_to_show)
        best_patches_indices = partitioned_indices[-number_of_extreme_patches_to_show:]

        return worst_patches_indices, best_patches_indices

    def get_extreme_patches_unwhitened(self, worst_patches_indices, best_patches_indices, only_inv_t: bool = False):
        if self.args.dmh.replace_embedding_with_regular_conv_relu:
            kernel = self.embedding[0].weight
        else:
            kernel = self.embedding.kernel

        all_patches = kernel.data.cpu().numpy()
        worst_patches = all_patches[worst_patches_indices]
        best_patches = all_patches[best_patches_indices]

        both_patches = np.concatenate([worst_patches, best_patches])
        both_patches_unwhitened = self.unwhiten_patches(both_patches, only_inv_t)

        worst_patches_unwhitened = both_patches_unwhitened[:len(worst_patches)]
        best_patches_unwhitened = both_patches_unwhitened[len(best_patches):]

        return worst_patches_unwhitened, best_patches_unwhitened

    def visualize_patches(self, n: int = 3):
        if self.bottle_neck is not None:
            bottleneck_weight = self.bottle_neck.weight.data.squeeze(dim=3).squeeze(dim=2)
            norms = torch.linalg.norm(bottleneck_weight, ord=2, dim=0).cpu().numpy()
        else:
            norms = np.random.default_rng().uniform(size=self.args.dmh.n_clusters).astype(np.float32)
        
        if n ** 2 >= len(norms) / 2:  # Otherwise we'll crash when asking for n**2 best/worth norms.
            n = math.floor(math.sqrt(len(norms) / 2))
        
        worst_patches_indices, best_patches_indices = self.get_extreme_patches_indices(norms, n ** 2)
        worst_patches_unwhitened, best_patches_unwhitened = self.get_extreme_patches_unwhitened(
            worst_patches_indices, best_patches_indices)
        worst_patches_whitened, best_patches_whitened = self.get_extreme_patches_unwhitened(
            worst_patches_indices, best_patches_indices, only_inv_t=True)

        # Normalize the values to be in [0, 1] for plotting (matplotlib just clips the values).
        for a in [worst_patches_unwhitened, best_patches_unwhitened, worst_patches_whitened, best_patches_whitened]:
            a[:] = (a - a.min()) / (a.max() - a.min())

        worst_patches_fig, worst_patches_axs = plt.subplots(n, n)
        best_patches_fig, best_patches_axs = plt.subplots(n, n)
        worst_patches_whitened_fig, worst_patches_whitened_axs = plt.subplots(n, n)
        best_patches_whitened_fig, best_patches_whitened_axs = plt.subplots(n, n)

        for k in range(n ** 2):
            i = k // n  # Row index
            j = k % n  # Columns index
            worst_patches_axs[i, j].imshow(worst_patches_unwhitened[k].transpose(1, 2, 0), vmin=0, vmax=1)
            best_patches_axs[i, j].imshow(best_patches_unwhitened[k].transpose(1, 2, 0), vmin=0, vmax=1)
            worst_patches_whitened_axs[i, j].imshow(worst_patches_whitened[k].transpose(1, 2, 0), vmin=0, vmax=1)
            best_patches_whitened_axs[i, j].imshow(best_patches_whitened[k].transpose(1, 2, 0), vmin=0, vmax=1)

            worst_patches_axs[i, j].axis('off')
            best_patches_axs[i, j].axis('off')
            worst_patches_whitened_axs[i, j].axis('off')
            best_patches_whitened_axs[i, j].axis('off')

        self.trainer.logger.experiment.log({'worst_patches': worst_patches_fig,
                                            'best_patches': best_patches_fig,
                                            'worst_patches_whitened': worst_patches_whitened_fig,
                                            'best_patches_whitened': best_patches_whitened_fig},
                                           step=self.trainer.global_step)
        plt.close('all')  # Avoid memory consumption

    def log_covariance_matrix(self, covariance_matrix: np.ndarray, name: str):
        labels = [f'{i:0>3}' for i in range(covariance_matrix.shape[0])]
        name = f'{name}_cov_matrix'
        self.logger.experiment.log(
            {name: wandb.plots.HeatMap(x_labels=labels, y_labels=labels, matrix_values=covariance_matrix)},
            step=self.trainer.global_step)
        self.logger.experiment.summary[f'{name}_ratio'] = self.get_diagonal_sum_vs_total_sum_ratio(covariance_matrix)

    def on_train_start(self):
        if self.input_channels == 3:
            self.visualize_patches()
        m = 64  # this is the maximal number of elements to take when calculating in covariance matrix
        if self.patches_covariance_matrix is not None:
            self.log_covariance_matrix(self.patches_covariance_matrix[:m, :m], name='patches')
        if self.args.dmh.use_whitening and (self.whitened_patches_covariance_matrix is not None):
            self.log_covariance_matrix(self.whitened_patches_covariance_matrix[:m, :m], name='whitened_patches')

    def on_train_end(self):
        if self.input_channels == 3:
            self.visualize_patches()


class ShufflePixels:
    def __init__(self, keep_rgb_triplets_intact: bool = True):
        self.keep_rgb_triplets_intact = keep_rgb_triplets_intact

    def __call__(self, img):
        assert img.shape == (3, 32, 32), "WTF is the shape of the input images?"
        start_dim = 1 if self.keep_rgb_triplets_intact else 0
        img_flat = torch.flatten(img, start_dim=start_dim)
        permutation = torch.randperm(img_flat.shape[-1])
        permuted_img_flat = img_flat[..., permutation]
        permuted_img = torch.reshape(permuted_img_flat, shape=img.shape)
        return permuted_img


class DataModule(LightningDataModule):

    @staticmethod
    def get_normalization_transform(plus_minus_one: bool = False, unit_gaussian: bool = False,
                                    n_channels: int = 3):
        if unit_gaussian:
            if n_channels != 3:
                raise NotImplementedError('Normalization for MNIST / FashionMNIST is not supported. ')
            normalization_values = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
        elif plus_minus_one:
            normalization_values = [(0.5, ) * n_channels] * 2  # times 2 because one is mean and one is std
        else:
            return None

        return Normalize(*normalization_values)

    @staticmethod
    def get_augmentations_transforms(random_flip: bool = False, random_crop: bool = False, spatial_size: int = 32):
        augmentations_transforms = list()

        if random_flip:
            augmentations_transforms.append(RandomHorizontalFlip())
        if random_crop:
            augmentations_transforms.append(RandomCrop(size=spatial_size, padding=4))

        return augmentations_transforms

    @staticmethod
    def get_transforms_lists(args: DataArgs):
        augmentations = DataModule.get_augmentations_transforms(args.random_horizontal_flip, args.random_crop,
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

    def get_dataset_class(self, dataset_name: str):
        if dataset_name == 'CIFAR10':
            return CIFAR10
        elif dataset_name == 'MNIST':
            return MNIST
        elif dataset_name == 'FashionMNIST':
            return FashionMNIST
        else:
            raise NotImplementedError(f'Dataset {dataset_name} is not implemented.')

    def __init__(self, args: DataArgs, batch_size: int, data_dir: str = "./data"):
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

    def prepare_data(self):
        for train_mode in [True, False]:
            self.dataset_class(self.data_dir, train=train_mode, download=True)

    def setup(self, stage: Optional[str] = None):
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
        return DataLoader(self.datasets['fit_aug'], batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return [
            DataLoader(self.datasets['validate_aug'], batch_size=self.batch_size, num_workers=4),
            DataLoader(self.datasets['validate_no_aug'], batch_size=self.batch_size, num_workers=4)
        ]

    def train_dataloader_no_aug(self):
        return DataLoader(self.datasets['fit_no_aug'], batch_size=self.batch_size, num_workers=4, shuffle=True)

    def train_dataloader_clean(self):
        return DataLoader(self.datasets['fit_clean'], batch_size=self.batch_size, num_workers=4, shuffle=True)


class LitVGG(pl.LightningModule):

    def __init__(self, args: Args):
        super(LitVGG, self).__init__()
        layers, _, _, features_output_dimension = get_blocks(configs[args.arch.model_name],
                                                             args.arch.dropout_prob,
                                                             args.arch.padding_mode)
        self.features = nn.Sequential(*layers)
        self.mlp = get_mlp(input_dim=features_output_dimension,
                           output_dim=N_CLASSES,
                           n_hidden_layers=args.arch.final_mlp_n_hidden_layers,
                           hidden_dim=args.arch.final_mlp_hidden_dim,
                           use_batch_norm=True,
                           organize_as_blocks=True)
        self.loss = torch.nn.CrossEntropyLoss()
        self.args: Args = args
        self.save_hyperparameters(args.flattened_dict())

        self.num_blocks = len(self.features) + len(self.mlp)

        # Apparently the Metrics must be an attribute of the LightningModule, and not inside a dictionary.
        # This is why we have to set them separately here and then the dictionary will map to the attributes.
        self.train_accuracy = tm.Accuracy()
        self.validate_accuracy = tm.Accuracy()
        self.validate_no_aug_accuracy = tm.Accuracy()
        self.accuracy = {RunningStage.TRAINING: [self.train_accuracy],
                         RunningStage.VALIDATING: [self.validate_accuracy, self.validate_no_aug_accuracy]}

        self.kernel_sizes: List[int] = self.get_kernel_sizes()  # For each convolution/pooling block
        self.shapes: List[tuple] = self.get_shapes()

    def forward(self, x: torch.Tensor):
        features = self.features(x)
        outputs = self.mlp(features)
        return outputs

    def shared_step(self, batch, stage: RunningStage, dataloader_idx: int = 0):
        inputs, labels = batch
        logits = self(inputs)
        loss = self.loss(logits, labels)

        self.accuracy[stage][dataloader_idx](logits, labels)

        name = stage.value if dataloader_idx == 0 else f'{stage.value}_no_aug'
        self.log(f'{name}_loss', loss)
        self.log(f'{name}_accuracy', self.accuracy[stage][dataloader_idx])

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, RunningStage.TRAINING)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        self.shared_step(batch, RunningStage.VALIDATING, dataloader_idx)

    def get_sub_model(self, i: int) -> nn.Sequential:
        if i < len(self.features):
            sub_model = self.features[:i]
        else:
            j = len(self.features) - i  # This is the index in the mlp
            sub_model = nn.Sequential(*(list(self.features) + list(self.mlp[:j])))

        return sub_model

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    self.args.opt.learning_rate,
                                    self.args.opt.momentum,
                                    weight_decay=self.args.opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.args.opt.learning_rate_decay_steps,
                                                         gamma=self.args.opt.learning_rate_decay_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_kernel_sizes(self):
        kernel_sizes = list()
        for i in range(len(self.features)):
            kernel_size = get_vgg_model_kernel_size(self, i)
            if isinstance(kernel_size, tuple):
                assert kernel_size[0] == kernel_size[1], "Only square patches are supported"
                kernel_size = kernel_size[0]
            kernel_sizes.append(kernel_size)
        return kernel_sizes

    @torch.no_grad()
    def get_shapes(self):
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

    def __init__(self, args: Args):
        super(LitMLP, self).__init__()
        self.args = args
        self.input_dim = args.data.n_channels * args.data.spatial_size ** 2
        self.output_dim = N_CLASSES
        self.n_hidden_layers = args.arch.final_mlp_n_hidden_layers
        self.hidden_dim = args.arch.final_mlp_hidden_dim
        self.mlp = get_mlp(self.input_dim, self.output_dim, self.n_hidden_layers, self.hidden_dim,
                           use_batch_norm=True, organize_as_blocks=True)
        self.loss = torch.nn.CrossEntropyLoss()

        self.num_blocks = len(self.mlp)

        # Apparently the Metrics must be an attribute of the LightningModule, and not inside a dictionary.
        # This is why we have to set them separately here and then the dictionary will map to the attributes.
        self.train_accuracy = tm.Accuracy()
        self.validate_accuracy = tm.Accuracy()
        self.validate_no_aug_accuracy = tm.Accuracy()
        self.accuracy = {RunningStage.TRAINING: [self.train_accuracy],
                         RunningStage.VALIDATING: [self.validate_accuracy, self.validate_no_aug_accuracy]}

    def forward(self, x: torch.Tensor):
        return self.mlp(x)

    def shared_step(self, batch, stage: RunningStage, dataloader_idx: int = 0):
        inputs, labels = batch
        logits = self(inputs)
        loss = self.loss(logits, labels)

        self.accuracy[stage][dataloader_idx](logits, labels)

        name = stage.value
        if dataloader_idx == 1:
            name += '_no_aug'

        self.log(f'{name}_loss', loss)
        self.log(f'{name}_accuracy', self.accuracy[stage][dataloader_idx])

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, RunningStage.TRAINING)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        self.shared_step(batch, RunningStage.VALIDATING, dataloader_idx)

    def get_sub_model(self, i: int) -> nn.Sequential:
        return self.mlp[:i]

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    self.args.opt.learning_rate,
                                    self.args.opt.momentum,
                                    weight_decay=self.args.opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.args.opt.learning_rate_decay_steps,
                                                         gamma=self.args.opt.learning_rate_decay_gamma)
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


class IntrinsicDimensionCalculator(Callback):

    def __init__(self, args: DMHArgs, minimal_distance: float = 1e-05):
        """
        Since the intrinsic-dimension calculation takes logarithm of the distances,
        if they are zero (or very small) it can cause numerical issues (NaN).
        """
        self.n: int = args.n_patches
        self.k1: int = args.k1
        self.k2: int = args.k2
        self.k3: int = 8 * self.k2  # This will be used to plot a graph of the intrinsic-dimension per k (until k3)
        self.estimate_dim_on_patches: bool = args.estimate_dim_on_patches
        self.estimate_dim_on_images: bool = args.estimate_dim_on_images
        self.shuffle_before_estimate: bool = args.shuffle_before_estimate
        self.minimal_distance = minimal_distance

        # Since the intrinsic-dimension calculation takes logarithm of the distances,
        # if they are zero (or very small) it can cause numerical issues (NaN).
        # The solution is to sample a bit more patches than requested,
        # and later we remove patches that are really close to one another,
        # and we want our final number of patches to be the desired one.
        ratio_to_extend_n = 1.5 if self.estimate_dim_on_patches else 1.01  # Lower probability to get similar images.
        self.n_extended = math.ceil(self.n * ratio_to_extend_n)

    @staticmethod
    def log_dim_per_k_graph(metrics, block_name, estimates):
        min_k = 5
        max_k = len(estimates) + 1
        df = pd.DataFrame(estimates[min_k - 1:], index=np.arange(min_k, max_k), columns=['k-th intrinsic-dimension'])
        df.index.name = 'k'
        metrics[f'{block_name}-int_dim_per_k'] = px.line(df)

    @staticmethod
    def log_singular_values(metrics, block_name, data):
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
        metrics[f'{block_name}-d_cov'] = np.where(variance_ratio['original_data'] > 0.95)[0][0]

        fig_args = dict(markers=True)
        metrics[f'{block_name}-singular_values'] = px.line(pd.DataFrame(singular_values), **fig_args)
        metrics[f'{block_name}-singular_values_ratio'] = px.line(pd.DataFrame(singular_values_ratio), **fig_args)
        metrics[f'{block_name}-variance_ratio'] = px.line(pd.DataFrame(variance_ratio), **fig_args)
        metrics[f'{block_name}-explained_variance_ratio'] = px.line(pd.DataFrame(explained_variance_ratio), **fig_args)
        metrics[f'{block_name}-reconstruction_errors'] = px.line(pd.DataFrame(reconstruction_errors), **fig_args)

    def log_final_estimate(self, metrics, estimates, extrinsic_dimension, block_name):
        estimate_mean_over_k1_to_k2 = torch.mean(estimates[self.k1:self.k2 + 1])
        intrinsic_dimension = estimate_mean_over_k1_to_k2.item()
        dimensions_ratio = intrinsic_dimension / extrinsic_dimension

        block_name = f'{block_name}-ext_dim_{extrinsic_dimension}'
        metrics.update({f'{block_name}-int_dim': intrinsic_dimension,
                        f'{block_name}-dim_ratio': dimensions_ratio})
        return intrinsic_dimension, dimensions_ratio

    def calc_int_dim_per_layer_on_dataloader(self, trainer, pl_module, dataloader, log_graphs: bool = False):
        """
        Given a VGG model, go over each block in it and calculates the intrinsic dimension of its input data.
        """

        log_graphs = log_graphs or (trainer.global_step == 0)

        metrics = dict()
        for i in range(pl_module.num_blocks):
            block_name = f'block_{i}'
            if self.estimate_dim_on_images or (i >= len(pl_module.kernel_sizes)):
                patch_size = -1
            else:
                patch_size = pl_module.kernel_sizes[i]
            patches = self.get_patches_not_too_close_to_one_another(dataloader, patch_size, pl_module.get_sub_model(i))

            estimates_matrix = get_estimates_matrix(patches, self.k3 if log_graphs else self.k2)
            estimates = torch.mean(estimates_matrix, dim=0)

            if log_graphs:
                IntrinsicDimensionCalculator.log_dim_per_k_graph(metrics, block_name, estimates.cpu().numpy())
                IntrinsicDimensionCalculator.log_singular_values(metrics, block_name, patches.cpu().numpy())

            mle_int_dim, ratio = self.log_final_estimate(metrics, estimates, patches.shape[1], block_name)
            logger.debug(f'epoch {trainer.current_epoch:0>2d} block {i:0>2d} '
                         f'mle_int_dim {mle_int_dim:.2f} ({100 * ratio:.2f}% of ext_sim {patches.shape[1]})')
        trainer.logger.experiment.log(metrics, step=trainer.global_step, commit=False)

    def get_patches_not_too_close_to_one_another(self, dataloader, patch_size, sub_model, device=None):
        patches = self.get_flattened_patches(dataloader, patch_size, sub_model, device)
        patches_to_keep_mask = self.get_patches_to_keep_mask(patches)
        patches = patches[patches_to_keep_mask]

        patches = patches[:self.n]  # This is done to get exactly (or up-to) n like the user requested

        return patches

    def get_flattened_patches(self, dataloader, kernel_size, sub_model, device=None):
        patches = sample_random_patches(dataloader, self.n_extended, kernel_size, sub_model)
        patches = patches.reshape(patches.shape[0], -1)  # a.k.a. flatten in NumPy

        if self.shuffle_before_estimate:
            patches = np.random.default_rng().permuted(patches, axis=1, out=patches)

        patches = patches.astype(np.float64)  # Increase accuracy of calculations.
        patches = torch.from_numpy(patches)

        if device is not None:
            patches = patches.to(device)

        return patches

    def get_patches_to_keep_mask(self, patches):
        distance_matrix = torch.cdist(patches, patches)
        small_distances_indices = torch.nonzero(torch.less(distance_matrix, self.minimal_distance))
        different_patches_mask = (small_distances_indices[:, 0] != small_distances_indices[:, 1])
        different_patches_close_indices_pairs = small_distances_indices[different_patches_mask]
        different_patches_close_indices = different_patches_close_indices_pairs.unique()
        patches_to_keep_mask = indices_to_mask(len(patches), different_patches_close_indices, negate=True)

        return patches_to_keep_mask

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


def initialize_model(args: Args, wandb_logger: WandbLogger):
    if args.arch.model_name.startswith('VGG'):
        model_class = LitVGG
    else:
        model_class = LitMLP

    if args.arch.use_pretrained:
        artifact = wandb_logger.experiment.use_artifact(args.arch.pretrained_path, type='model')
        artifact_dir = artifact.download()
        model = model_class.load_from_checkpoint(str(Path(artifact_dir) / "model.ckpt"), args=args)
    else:
        model = model_class(args)

    return model


def initialize_wandb_logger(args: Args):
    return WandbLogger(project='thesis',
                       config=args.flattened_dict(),
                       name=args.env.wandb_run_name,
                       log_model=True)


def initialize_trainer(args: Args, wandb_logger: WandbLogger):
    checkpoint_callback = ModelCheckpoint(monitor='validate_no_aug_accuracy/dataloader_idx_1', mode='max')
    model_summary_callback = ModelSummary(max_depth=3)
    callbacks = [checkpoint_callback, model_summary_callback]
    trainer_kwargs = dict(gpus=[args.env.device_num]) if args.env.is_cuda else dict()
    if args.env.debug:
        trainer_kwargs.update({f'limit_{t}_batches': 3 for t in ['train', 'val']})
        trainer_kwargs.update({'log_every_n_steps': 1})
    if args.dmh.estimate_intrinsic_dimension:
        callbacks.append(IntrinsicDimensionCalculator(args.dmh))
    trainer = pl.Trainer(logger=wandb_logger, callbacks=callbacks, max_epochs=args.opt.epochs,
                         **trainer_kwargs)
    return trainer


class ImitatorKNN:  # TODO inherit from pl.LightningModule to enable saving checkpoint of the model
    @torch.no_grad()
    def __init__(self, teacher: LitVGG, dataloader: DataLoader, args: DMHArgs):
        self.teacher: LitVGG = teacher
        self.dataloader: DataLoader = dataloader
        self.n_patches: int = args.n_patches
        self.n_clusters: int = args.n_clusters
        self.random_patches: bool = args.random_patches  # TODO name was changed (PyCharm did not refactor well)

        self.centroids_outputs: List[np.ndarray] = list()
        self.kmeans: List[faiss.Kmeans] = list()

        for i, block in enumerate(teacher.features):
            patches = sample_random_patches(self.dataloader, self.n_patches,  # TODO whiten / normalize the patches?
                                            self.teacher.kernel_sizes[i], self.teacher.get_sub_model(i),
                                            random_patches=args.random_patches)
            patches_flat = patches.reshape(patches.shape[0], -1)
            kmeans = faiss.Kmeans(d=patches_flat.shape[1], k=self.n_clusters, verbose=True)
            kmeans.train(patches_flat)
            centroids = kmeans.centroids.reshape(-1, *patches.shape[1:])

            if isinstance(block, nn.Sequential):
                conv_layer = block[0]
                centroids = torch.from_numpy(centroids).to(conv_layer.weight.device)
                centroids_outputs = F.conv2d(
                    centroids, conv_layer.weight.detach(), conv_layer.bias.detach()
                ).squeeze(dim=-1).squeeze(dim=-1).cpu().numpy()
            else:  # block is a MaxPool layer
                centroids_outputs = centroids.max(axis=(-2, -1))

            self.centroids_outputs.append(centroids_outputs)
            self.kmeans.append(kmeans)

    def forward(self, x: torch.Tensor):
        """
        Returns the output of all blocks
        """
        intermediate_errors: List[np.ndarray] = list()
        for i, block in enumerate(self.teacher.features):
            if isinstance(block, nn.Sequential):
                # x might be on GPU, but our knn-imitator works on CPU only (since it requires more memory than on GPU)
                x_cpu = x.cpu()

                conv_layer = block[0]
                conv_output = conv_layer(x)
                conv_output_size = tuple(conv_output.shape[-2:])
                conv_kwargs = {k: getattr(conv_layer, k) for k in ['kernel_size', 'dilation', 'padding', 'stride']}

                x_unfolded = F.unfold(x_cpu, **conv_kwargs)
                batch_size, patch_dim, n_patches = x_unfolded.shape

                # Transpose from (N, C*H*W, M) to (N, M, C*H*W) and then reshape to (N*M, C*H*W) to have collection of vectors
                # Also make contiguous in memory (required by function kmeans.search).
                x_unfolded = x_unfolded.transpose(dim0=1, dim1=2).flatten(start_dim=0, end_dim=1).contiguous()

                # ##########################################################################
                # # This piece of code verifies the unfold mechanism by
                # # simulating the conv layer using matrix multiplication.
                # ##########################################################################
                # conv_weight = conv_layer.weight.flatten(start_dim=1).T
                # conv_output_mm = torch.mm(x_unfolded, conv_weight)
                # conv_output_mm = conv_output_mm.reshape(batch_size, n_patches, -1)
                # conv_output_mm = conv_output_mm.transpose(dim0=1, dim1=2)
                # conv_output_mm = conv_output_mm.reshape(batch_size, -1, *conv_output_size)
                # if not torch.allclose(conv_output, conv_output_mm):
                #     print(f'mean = {torch.mean(torch.abs(conv_output - conv_output_mm))}')
                #     print(f'max = {torch.max(torch.abs(conv_output - conv_output_mm))}')
                # ##########################################################################

                distances, indices = self.kmeans[i].assign(x_unfolded.numpy())
                x_unfolded_outputs = self.centroids_outputs[i][indices]
                x_unfolded_outputs = torch.from_numpy(x_unfolded_outputs)
                x_unfolded_outputs = x_unfolded_outputs.reshape(batch_size, n_patches, -1)
                x_unfolded_outputs = x_unfolded_outputs.transpose(dim0=1, dim1=2)
                x_output = x_unfolded_outputs.reshape(batch_size, -1, *conv_output_size)
                x_output = x_output.to(conv_output.device)

                intermediate_errors.append(torch.mean(torch.abs(x_output - conv_output),
                                                      dim=tuple(range(1, x_output.ndim))).cpu().numpy())

                # Run the rest of the block (i.e. BatchNorm->ReLU) on the calculated output
                x = block[1:](x_output)
            else:  # block is a MaxPool layer
                # TODO Do we want to simulate MaxPool with kNN as well?
                # No error since we don't simulate the MaxPool layers.
                intermediate_errors.append(np.zeros(x.shape[0], dtype=np.float32))
                x = block(x)

        logits = self.teacher.mlp(x)

        return logits, np.array(intermediate_errors)

    def __call__(self, x: torch.Tensor):
        """
        Returns the final prediction of the model (output of the last layer)
        """
        logits, intermediate_errors = self.forward(x)
        return logits

    @torch.no_grad()
    def evaluate(self, dataloader):
        corrects_sum = 0
        total_intermediate_errors = np.zeros(shape=(len(self.teacher.features), 0), dtype=np.float32)

        for inputs, labels in dataloader:
            logits, intermediate_errors = self.forward(inputs)
            _, predictions = torch.max(logits, dim=1)
            total_intermediate_errors = np.concatenate([total_intermediate_errors, intermediate_errors], axis=1)
            corrects_sum += torch.sum(torch.eq(predictions, labels.data)).item()

        accuracy = 100 * (corrects_sum / len(dataloader.dataset))

        return total_intermediate_errors, accuracy


def evaluate_knn_imitator(imitator: ImitatorKNN, datamodule: pl.LightningDataModule, wandb_logger: WandbLogger):
    intermediate_errors, accuracy = imitator.evaluate(dataloader=datamodule.val_dataloader()[1])

    conv_blocks_indices = [i for i, block in enumerate(imitator.teacher.features) if isinstance(block, nn.Sequential)]
    wandb_logger.experiment.log({'knn_imitator_error': ff.create_distplot(
        intermediate_errors[conv_blocks_indices],
        group_labels=[f'block_{i}_error' for i in conv_blocks_indices],
        show_hist=False
    )})

    for i, error in enumerate(intermediate_errors.mean(axis=1)):
        wandb_logger.experiment.summary[f'knn_block_{i}_imitator_mean_error'] = error
    wandb_logger.experiment.summary['knn_imitator_accuracy'] = accuracy


def initialize_datamodule(args: DataArgs, batch_size: int):
    datamodule = DataModule(args, batch_size)
    datamodule.prepare_data()
    datamodule.setup(stage='fit')
    datamodule.setup(stage='validate')

    return datamodule


def get_pre_model(wandb_logger, args):
    artifact = wandb_logger.experiment.use_artifact(args.arch.pretrained_path, type='model')
    artifact_dir = artifact.download()
    checkpoint_path = str(Path(artifact_dir) / "model.ckpt")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    pre_model_args: Args = get_args_from_flattened_dict(
        Args, checkpoint['hyper_parameters'],
        excluded_categories=['env']  # Ignore environment args, such as GPU (which will raise error if on CPU).
    )
    pre_model_args.env = args.env
    pre_model = LocallyLinearNetwork.load_from_checkpoint(
        checkpoint_path, map_location=torch.device('cpu'), args=pre_model_args)
    pre_model.requires_grad_(False)
    pre_model.eval()
    pre_model.embedding_mode()
    pre_model.linear = None  # we don't need the linear layer predicting the logits anymore.

    return pre_model


def main():
    args = get_args(args_class=Args)
    datamodule = initialize_datamodule(args.data, args.opt.batch_size)
    wandb_logger = initialize_wandb_logger(args)
    model = initialize_model(args, wandb_logger)
    configure_logger(args.env.path, print_sink=sys.stdout, level='DEBUG')  # if args.env.debug else 'INFO')

    if args.dmh.train_locally_linear_network:
        pre_model = None if (args.dmh.depth == 1) else get_pre_model(wandb_logger, args)
        model = LocallyLinearNetwork(args, pre_model)
        if not args.dmh.replace_embedding_with_regular_conv_relu:
            model.to(args.env.device)
            model.calculate_embedding_from_data(datamodule.train_dataloader_clean())
            model.cpu()
        wandb_logger.watch(model, log='all')
        trainer = initialize_trainer(args, wandb_logger)
        trainer.fit(model, datamodule=datamodule)
    elif not args.arch.use_pretrained:
        wandb_logger.watch(model, log='all')
        trainer = initialize_trainer(args, wandb_logger)
        trainer.fit(model, datamodule=datamodule)

    if args.dmh.imitate_with_knn:
        # TODO do we want to set-up knn-imitator on data without augmentations?
        imitator = ImitatorKNN(model, datamodule.train_dataloader(), args.dmh)
        evaluate_knn_imitator(imitator, datamodule, wandb_logger)

    if args.dmh.imitate_with_locally_linear_model:
        imitator = LocallyLinearImitatorVGG(model, datamodule, args)
        wandb_logger.watch(imitator, log='all')
        trainer = initialize_trainer(args, wandb_logger)
        trainer.fit(imitator, datamodule=datamodule)


if __name__ == '__main__':
    main()
