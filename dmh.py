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
import wandb

from loguru import logger
from typing import Optional, List, Tuple
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, RandomCrop, RandomHorizontalFlip, Normalize, Compose
from pytorch_lightning import LightningDataModule, Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.states import RunningStage

from consts import CIFAR10_IMAGE_SIZE, N_CLASSES
from patches import sample_random_patches
from schemas.data import DataArgs
from schemas.dmh import Args, IntDimEstArgs, ImitationArgs
from utils import configure_logger, get_args, power_minus_1, get_mlp, get_dataloaders
from vgg import get_vgg_model_kernel_size, get_blocks, configs


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
                         for stage in (RunningStage.TRAINING, RunningStage.VALIDATING)
                         for aug in ('aug', 'no_aug')}

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[RunningStage] = None):
        if stage is None:
            return

        for s in (RunningStage.TRAINING, RunningStage.VALIDATING):
            for aug in ['aug', 'no_aug']:
                k = f'{s}_{aug}'
                if self.datasets[k] is None:
                    self.datasets[k] = CIFAR10(self.data_dir,
                                               train=(s == RunningStage.TRAINING),
                                               transform=self.transforms[aug])

    def train_dataloader(self):
        return DataLoader(self.datasets[f'{RunningStage.TRAINING}_aug'], shuffle=True,
                          batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return [
            DataLoader(self.datasets[f'{RunningStage.VALIDATING}_aug'], batch_size=self.batch_size, num_workers=4),
            DataLoader(self.datasets[f'{RunningStage.VALIDATING}_no_aug'], batch_size=self.batch_size, num_workers=4)
        ]


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

        name = str(stage)
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

        name = str(stage)
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

    def __init__(self, args: IntDimEstArgs, minimal_distance: float = 1e-05):
        """
        Since the intrinsic-dimension calculation takes logarithm of the distances,
        if they are zero (or very small) it can cause numerical issues (NaN).
        """
        self.n: int = args.n
        self.k1: int = args.k1
        self.k2: int = args.k2
        self.k3: int = 8 * self.k2  # This will be used to plot a graph of the intrinsic-dimension per k (until k3)
        self.estimate_dim_on_patches: bool = args.estimate_dim_on_patches
        self.estimate_dim_on_images: bool = args.estimate_dim_on_images
        self.log_graphs: bool = args.log_graphs
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
        u, s, v = np.linalg.svd(covariance_matrix, hermitian=True)
        rotated_data = centered_data @ u
        whitened_data = rotated_data / (np.sqrt(s) + epsilon)
        return whitened_data

    def log_singular_values(self, metrics, block_name, data):
        n_samples, n_features = data.shape
        data_dict = {'original_data': data,
                     'normalized_data': self.normalize_data(data),
                     'whitened_data': self.whiten_data(data),
                     'random_data': np.random.default_rng().multivariate_normal(mean=np.zeros(n_features),
                                                                                cov=np.eye(n_features),
                                                                                size=n_samples)}

        singular_values = {k: np.linalg.svd(data_dict[k], compute_uv=False) for k in data_dict.keys()}
        singular_values_ratio = {k: (singular_values[k] / singular_values[k][0]) for k in data_dict.keys()}

        # Inspiration taken from sklearn.decomposition._pca.PCA._fit_full
        explained_variance_ratio = dict()
        for k in data_dict.keys():
            explained_variance = (singular_values[k] ** 2) / (n_samples - 1)
            explained_variance_ratio[k] = explained_variance / explained_variance.sum()

        fig_args = dict(markers=True)
        metrics[f'{block_name}-singular_values'] = px.line(pd.DataFrame(singular_values), **fig_args)
        metrics[f'{block_name}-singular_values_ratio'] = px.line(pd.DataFrame(singular_values_ratio), **fig_args)
        metrics[f'{block_name}-explained_variance_ratio'] = px.line(pd.DataFrame(explained_variance_ratio), **fig_args)

    def log_final_estimate(self, metrics, estimates, extrinsic_dimension, block_name):
        estimate_mean_over_k1_to_k2 = torch.mean(estimates[self.k1:self.k2 + 1])
        intrinsic_dimension = estimate_mean_over_k1_to_k2.item()
        dimensions_ratio = intrinsic_dimension / extrinsic_dimension

        block_name = f'{block_name}-ext_dim_{extrinsic_dimension}'
        metrics.update({f'{block_name}-int_dim': intrinsic_dimension,
                        f'{block_name}-dim_ratio': dimensions_ratio})
        return intrinsic_dimension, dimensions_ratio

    def calc_int_dim_per_layer_on_dataloader(self, trainer, pl_module, dataloader, name: str = ''):
        """
        Given a VGG model, go over each block in it and calculates the intrinsic dimension of its input data.
        """
        metrics = dict()
        for i in range(pl_module.num_blocks):
            block_name = f'{name}-block_{i}' if len(name) > 0 else f'block_{i}'
            if self.estimate_dim_on_images or (i >= len(pl_module.kernel_sizes)):
                patch_size = -1
            else:
                patch_size = pl_module.kernel_sizes[i]
            patches = self.get_patches_not_too_close_to_one_another(dataloader, patch_size, pl_module.get_sub_model(i))

            estimates_matrix = get_estimates_matrix(patches, self.k3 if self.log_graphs else self.k2)
            estimates = torch.mean(estimates_matrix, dim=0)

            if self.log_graphs:
                self.log_dim_per_k_graph(metrics, block_name, estimates.cpu().numpy())
                self.log_singular_values(metrics, block_name, patches.cpu().numpy())

            int_dim, ratio = self.log_final_estimate(metrics, estimates, patches.shape[1], block_name)
            logger.debug(f'epoch {trainer.current_epoch:0>2d} block {i:0>2d} '
                         f'int-dim {int_dim:.2f} ({100 * ratio:.2f}% of ext-sim {patches.shape[1]})')
        trainer.logger.experiment.log(metrics, step=trainer.global_step, commit=False)

    def get_patches_not_too_close_to_one_another(self, dataloader, patch_size, sub_model, device=None):
        patches = self.get_flattened_patches(dataloader, patch_size, sub_model, device)
        patches_to_keep_mask = self.get_patches_to_keep_mask(patches)
        patches = patches[patches_to_keep_mask]

        patches = patches[:self.n]  # This is done to get exactly n like the user requested.

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

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Calculates the intrinsic-dimension of the model, both on the training-data and test-data.
        """
        validation_data_without_augmentations = trainer.request_dataloader(RunningStage.VALIDATING)[1]
        self.calc_int_dim_per_layer_on_dataloader(trainer, pl_module, dataloader=validation_data_without_augmentations)


def initialize_model(args: Args):
    if args.arch.model_name.startswith('VGG'):
        return LitVGG(args)
    else:
        return LitMLP(args)


def initialize_wandb_logger(args: Args, model: nn.Module):
    wandb_logger = WandbLogger(project='thesis', config=args.flattened_dict(), name=args.env.wandb_run_name)
    wandb_logger.watch(model)

    return wandb_logger


def initialize_trainer(args: Args, model: nn.Module):
    wandb_logger = initialize_wandb_logger(args, model)
    callbacks = list()
    trainer_kwargs = dict(gpus=[args.env.device_num]) if args.env.is_cuda else dict()
    if args.env.debug:
        trainer_kwargs.update({f'limit_{t}_batches': 3 for t in ['train', 'val']})
        trainer_kwargs.update({'log_every_n_steps': 1})
    if args.int_dim_est.estimate_intrinsic_dimension:
        callbacks.append(IntrinsicDimensionCalculator(args.int_dim_est))
    trainer = pl.Trainer(logger=wandb_logger, callbacks=callbacks, max_epochs=args.opt.epochs,
                         **trainer_kwargs)
    return trainer


class ImitatorKNN:
    @torch.no_grad()
    def __init__(self, teacher: LitVGG, dataloader: DataLoader, args: ImitationArgs):
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

    @ torch.no_grad()
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


def main():
    args = get_args(args_class=Args)

    model = initialize_model(args)
    trainer = initialize_trainer(args, model)
    datamodule = CIFAR10DataModule(args.data, args.opt.batch_size)

    logger_kwargs = dict(level='DEBUG') if args.env.debug else dict()
    configure_logger(args.env.path, print_sink=model.print, **logger_kwargs)
    trainer.fit(model, datamodule=datamodule)

    if args.imitation.imitate_with_knn:
        # TODO do we want to set-up knn-imitator on data without augmentations?
        imitator = ImitatorKNN(model, datamodule.train_dataloader(), args.imitation)
        intermediate_errors, accuracy = imitator.evaluate(dataloader=datamodule.val_dataloader()[1])

        conv_blocks_indices = [i for i, block in enumerate(model.features) if isinstance(block, nn.Sequential)]
        trainer.logger.experiment.log({'knn_imitator_error': ff.create_distplot(
            intermediate_errors[conv_blocks_indices],
            group_labels=[f'block_{i}_error' for i in conv_blocks_indices],
            show_hist=False
        )})

        for i, error in enumerate(intermediate_errors.mean(axis=1)):
            wandb.summary[f'knn_block_{i}_imitator_mean_error'] = error
        wandb.summary['knn_imitator_accuracy'] = accuracy


if __name__ == '__main__':
    main()
