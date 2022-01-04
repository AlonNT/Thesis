import copy
import math
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
from typing import Optional, List, Union
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, RandomCrop, RandomHorizontalFlip, Normalize, Compose
from pytorch_lightning import LightningDataModule, Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

from consts import CIFAR10_IMAGE_SIZE, N_CLASSES, CIFAR10_IN_CHANNELS
from patches import sample_random_patches
from schemas.data import DataArgs
from schemas.dmh import Args, DMHArgs
from utils import configure_logger, get_args, power_minus_1, get_mlp, get_dataloaders
from vgg import get_vgg_model_kernel_size, get_blocks, configs


class KNearestPatchesEmbedding(nn.Module):
    """
    Calculating the k-nearest-neighbors is implemented as convolution with bias, as was done in
    The Unreasonable Effectiveness of Patches in Deep Convolutional Kernels Methods
    (https://arxiv.org/pdf/2101.07528.pdf)
    Details can be found in Appendix B (page 13).
    """

    def __init__(self,
                 patches: np.ndarray,
                 k: int,
                 up_to_k: bool = True,
                 padding: int = 0,
                 requires_grad: bool = False,
                 random_embedding: bool = False):
        super(KNearestPatchesEmbedding, self).__init__()

        self.k = k
        self.up_to_k = up_to_k
        self.padding = padding

        if random_embedding:
            out_channels, in_channels, kernel_height, kernel_width = patches.shape
            assert kernel_height == kernel_width, "the patches should be square"
            tmp_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_height, padding=self.padding)
            kernel_np = tmp_conv.weight.data.cpu().numpy().copy()
            bias_np = tmp_conv.bias.data.cpu().numpy().copy()
        else:
            kernel_np = -1 * patches
            bias_np = 0.5 * np.square(np.linalg.norm(patches.reshape(patches.shape[0], -1), axis=1))

        self.kernel = nn.Parameter(torch.Tensor(kernel_np), requires_grad=requires_grad)
        self.bias = nn.Parameter(torch.Tensor(bias_np), requires_grad=requires_grad)

    def forward(self, images):
        # In every spatial location ij, we'll have a vector containing the squared distances to all the patches.
        # Note that it's not really the squared distance, but the squared distance minus the squared-norm of the
        # input patch in that location, but minimizing this value will minimize the distance
        # (since the norm of the input patch is the same among all patches in the bank).
        distances = F.conv2d(images, self.kernel, self.bias, padding=self.padding)
        values, indices = distances.kthvalue(self.k, dim=1, keepdim=True)
        if self.up_to_k:
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
        self.k_nearest_neighbors_embedding = KNearestPatchesEmbedding(patches, k, padding=padding)

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

    def __init__(self, args: Args, datamodule: pl.LightningDataModule):
        super(LocallyLinearNetwork, self).__init__()
        self.args: Args = args
        self.save_hyperparameters(args.flattened_dict())

        self.dataloader = datamodule.train_dataloader()  # TODO do we want a different dataloader (maybe no aug)?
        self.embedding = KNearestPatchesEmbedding(self.get_clustered_patches(),
                                                  self.args.dmh.k,
                                                  self.args.dmh.up_to_k,
                                                  requires_grad=self.args.dmh.learnable_embedding,
                                                  random_embedding=self.args.dmh.random_embedding)

        # This convolution layer can be thought of as linear function per patch in the patch-dict,
        # each function is from (in_channels x kernel_size^2) to the reals.
        self.conv = None if (not self.args.dmh.use_conv) else nn.Conv2d(in_channels=CIFAR10_IN_CHANNELS,
                                                                        out_channels=self.args.dmh.n_clusters,
                                                                        kernel_size=self.args.dmh.patch_size)

        embedding_spatial_size = CIFAR10_IMAGE_SIZE - self.args.dmh.patch_size + 1
        intermediate_n_features = self.args.dmh.n_clusters * (embedding_spatial_size ** 2)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=intermediate_n_features, out_features=N_CLASSES)

        self.loss = nn.CrossEntropyLoss()

        # Apparently the Metrics must be an attribute of the LightningModule, and not inside a dictionary.
        # This is why we have to set them separately here and then the dictionary will map to the attributes.
        self.train_accuracy = tm.Accuracy()
        self.validate_accuracy = tm.Accuracy()
        self.validate_no_aug_accuracy = tm.Accuracy()
        self.accuracy = {RunningStage.TRAINING: [self.train_accuracy],
                         RunningStage.VALIDATING: [self.validate_accuracy, self.validate_no_aug_accuracy]}

    def get_clustered_patches(self):
        patches = sample_random_patches(self.dataloader, self.args.dmh.n_patches, self.args.dmh.patch_size,
                                        random_patches=self.args.dmh.random_patches)
        patches_flat = patches.reshape(patches.shape[0], -1)
        kmeans = faiss.Kmeans(d=patches_flat.shape[1], k=self.args.dmh.n_clusters, verbose=True)
        kmeans.train(patches_flat)
        centroids = kmeans.centroids.reshape(-1, *patches.shape[1:])
        return centroids

    def forward(self, x: torch.Tensor):
        features = self.embedding(x)
        if self.args.dmh.use_conv:
            features *= self.conv(x)
        features_flat = self.flatten(features)
        logits = self.linear(features_flat)
        return logits

    def shared_step(self, batch, stage: RunningStage, dataloader_idx=0):
        x, labels = batch  # Note that we do not use the labels for training, only for logging training accuracy.
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
        optimizer = torch.optim.SGD(self.parameters(),  # Explicitly optimize only the imitators parameters
                                    self.args.opt.learning_rate,
                                    self.args.opt.momentum,
                                    weight_decay=self.args.opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.args.opt.learning_rate_decay_steps,
                                                         gamma=self.args.opt.learning_rate_decay_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


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


class CIFAR10DataModule(LightningDataModule):

    @staticmethod
    def get_normalization_transform(plus_minus_one: bool = False, unit_gaussian: bool = False):
        if unit_gaussian:
            normalization_values = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
        elif plus_minus_one:
            normalization_values = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
        else:
            return list()

        return Normalize(*normalization_values)

    @staticmethod
    def get_augmentations_transforms(random_flip: bool = False, random_crop: bool = False):
        augmentations_transforms = list()

        if random_flip:
            augmentations_transforms.append(RandomHorizontalFlip())
        if random_crop:
            augmentations_transforms.append(RandomCrop(size=32, padding=4))

        return augmentations_transforms

    @staticmethod
    def get_transforms_lists(args: DataArgs):
        augmentations = CIFAR10DataModule.get_augmentations_transforms(args.random_horizontal_flip, args.random_crop)
        normalization = CIFAR10DataModule.get_normalization_transform(args.normalization_to_plus_minus_one,
                                                                      args.normalization_to_unit_gaussian)

        post_transforms = [ShufflePixels(args.keep_rgb_triplets_intact)] if args.shuffle_images else list()
        transforms_list_no_aug = [ToTensor()] + [normalization] + post_transforms
        transforms_list_with_aug = augmentations + [ToTensor()] + [normalization] + post_transforms

        return transforms_list_no_aug, transforms_list_with_aug

    def __init__(self, args: DataArgs, batch_size: int, data_dir: str = "./data"):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size

        transforms_list_no_aug, transforms_list_with_aug = CIFAR10DataModule.get_transforms_lists(args)
        self.transforms = {'aug': Compose(transforms_list_with_aug), 'no_aug': Compose(transforms_list_no_aug)}
        self.datasets = {f'{stage}_{aug}': None
                         for stage in ('fit', 'validate')
                         for aug in ('aug', 'no_aug')}

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage is None:
            return

        for s in ('fit', 'validate'):
            for aug in ('aug', 'no_aug'):
                k = f'{s}_{aug}'
                if self.datasets[k] is None:
                    self.datasets[k] = CIFAR10(self.data_dir, train=(s == 'fit'), transform=self.transforms[aug])

    def train_dataloader(self):
        return DataLoader(self.datasets['fit_aug'], batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return [
            DataLoader(self.datasets['validate_aug'], batch_size=self.batch_size, num_workers=4),
            DataLoader(self.datasets['validate_no_aug'], batch_size=self.batch_size, num_workers=4)
        ]

    def train_dataloader_no_aug(self):
        return DataLoader(self.datasets['fit'], batch_size=self.batch_size, num_workers=4, shuffle=True)


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
        self.input_dim = 3 * CIFAR10_IMAGE_SIZE ** 2
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
        stop = 'here'
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
    def normalize_data(data, epsilon=1e-05):
        centered_data = data - data.mean(axis=0)
        normalized_data = centered_data / (centered_data.std(axis=0) + epsilon)
        return normalized_data

    @staticmethod
    def whiten_data(data, epsilon=1e-05):
        centered_data = data - data.mean(axis=0)
        covariance_matrix = (1 / centered_data.shape[0]) * (centered_data.T @ centered_data)
        u, s, v_t = np.linalg.svd(covariance_matrix, hermitian=True)
        rotated_data = centered_data @ u
        whitened_data = rotated_data / (np.sqrt(s) + epsilon)
        return whitened_data

    def log_singular_values(self, metrics, block_name, data):
        n, d = data.shape
        data_dict = {
            'original_data': data,
            'normalized_data': self.normalize_data(data),
            'whitened_data': self.whiten_data(data),
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
                self.log_dim_per_k_graph(metrics, block_name, estimates.cpu().numpy())
                self.log_singular_values(metrics, block_name, patches.cpu().numpy())

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
        self.random_patches: bool = args.random_patches

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

                # Transpose to (N, M, C*H*W) and then reshape to (N*M, C*H*W) to have collection of vectors
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
    datamodule = CIFAR10DataModule(args, batch_size)
    datamodule.prepare_data()
    datamodule.setup(stage='fit')
    datamodule.setup(stage='validate')

    return datamodule


def main():
    args = get_args(args_class=Args)
    datamodule = initialize_datamodule(args.data, args.opt.batch_size)
    wandb_logger = initialize_wandb_logger(args)
    model = initialize_model(args, wandb_logger)
    configure_logger(args.env.path, print_sink=model.print, level='DEBUG')  # if args.env.debug else 'INFO')

    if args.dmh.train_locally_linear_network:
        model = LocallyLinearNetwork(args, datamodule)
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
