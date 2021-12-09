import math
from typing import Optional, List
import wandb
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from pytorch_lightning import LightningDataModule, Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.states import RunningStage
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, RandomCrop, RandomHorizontalFlip, Normalize, Compose

from consts import CLASSES
from patches import sample_random_patches
from schemas.data import DataArgs
from schemas.dmh import Args
from utils import configure_logger, get_args, log_args, power_minus_1, get_mlp
from vgg import get_vgg_model_kernel_size, get_blocks, configs


class CIFAR10DataModule(LightningDataModule):
    @staticmethod
    def get_normalization_transform(plus_minus_one: bool = False, unit_gaussian: bool = False):
        if unit_gaussian:
            normalization_values = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
        elif plus_minus_one:
            normalization_values = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
        else:
            return list()

        return [Normalize(*normalization_values)]

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
        normalizations = CIFAR10DataModule.get_normalization_transform(args.normalization_to_plus_minus_one,
                                                                       args.normalization_to_unit_gaussian)

        transforms_list_no_aug = [ToTensor()] + normalizations
        transforms_list_with_aug = augmentations + [ToTensor()] + normalizations

        return transforms_list_no_aug, transforms_list_with_aug

    def __init__(self, args: DataArgs, batch_size: int = 64, data_dir: str = "./data"):
        super().__init__()
        transforms_list_no_aug, transforms_list_with_aug = CIFAR10DataModule.get_transforms_lists(args)
        self.transforms = {'aug': Compose(transforms_list_with_aug), 'no_aug': Compose(transforms_list_no_aug)}
        self.datasets = {f'{stage}_{aug}': None
                         for stage in (RunningStage.TRAINING, RunningStage.VALIDATING)
                         for aug in ('aug', 'no_aug')}
        self.data_dir = data_dir
        self.batch_size = batch_size

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
                          batch_size=self.batch_size, num_workers=3)

    def val_dataloader(self):
        return [
            DataLoader(self.datasets[f'{RunningStage.VALIDATING}_aug'], batch_size=self.batch_size, num_workers=3),
            DataLoader(self.datasets[f'{RunningStage.VALIDATING}_no_aug'], batch_size=self.batch_size, num_workers=3)
        ]


class LitVGG(pl.LightningModule):
    # log_kwargs = dict(on_step=True, on_epoch=True, prog_bar=True, logger=True)
    log_kwargs = dict()

    def __init__(self, args: Args):
        super(LitVGG, self).__init__()
        layers, _, _, features_output_dimension = get_blocks(configs[args.arch.model_name],
                                                             args.arch.dropout_prob,
                                                             args.arch.padding_mode)
        self.features = nn.Sequential(*layers)
        self.mlp = get_mlp(input_dim=features_output_dimension,
                           output_dim=len(CLASSES),
                           n_hidden_layers=args.arch.final_mlp_n_hidden_layers,
                           hidden_dim=args.arch.final_mlp_hidden_dim)
        self.loss = torch.nn.CrossEntropyLoss()
        self.args: Args = args
        self.save_hyperparameters(args.flattened_dict())

        # Apparently the Metrics must be an attribute of the LightningModule, and not inside a dictionary.
        # This is why we have to set them separately here and then the dictionary will map to the attributes.
        self.train_accuracy = tm.Accuracy()
        self.validate_accuracy = tm.Accuracy()
        self.validate_no_aug_accuracy = tm.Accuracy()
        self.accuracy = {RunningStage.TRAINING: [self.train_accuracy],
                         RunningStage.VALIDATING: [self.validate_accuracy, self.validate_no_aug_accuracy]}

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

        self.log(f'{name}_loss', loss, **LitVGG.log_kwargs)
        self.log(f'{name}_accuracy', self.accuracy[stage][dataloader_idx], **LitVGG.log_kwargs)

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


def get_estimates_matrix(data: torch.Tensor, k: int):
    """Calculates a matrix containing the intrinsic-dimension estimators.

    The ij-th cell contains the j-th estimate for the i-th data-point.
    In the notation of the paper below it's $\\hat(m)_j(x_i)$.

    See `Maximum Likelihood Estimation of Intrinsic Dimension
    <https://papers.nips.cc/paper/2004/file/74934548253bcab8490ebd74afed7031-Paper.pdf>`_
    """
    assert data.ndim == 2, f"data has shape {tuple(data.shape)}, expected (n, d) i.e. n d-dimensional vectors. "

    if k > data.shape[0]:
        print(f"Number of data-points is {data.shape[0]} and k={k} should be smaller. ")
        k = data.shape[0] - 1
        print(f"k was changed to {k}")

    distance_matrix = torch.cdist(data, data)

    distances, _ = torch.topk(distance_matrix, k=1 + k, largest=False)
    distances = distances[:, 1:]  # Remove the 1st column corresponding to the (zero) distance between item and itself.
    log_distances = torch.log(distances)
    log_distances_cumsum = torch.cumsum(log_distances, dim=1)
    log_distances_cummean = torch.divide(log_distances_cumsum, torch.arange(start=1, end=log_distances.shape[1] + 1))
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
    mask = torch.scatter(torch.zeros(n, dtype=torch.bool), dim=0, index=indices, value=1)
    if negate:
        mask = torch.bitwise_not(mask)
    return mask


class IntrinsicDimensionCalculator(Callback):

    def __init__(self, n_patches: int, k1: int, k2: int, plot_graphs: bool,
                 minimal_distance_between_patches: float = 1e-05):
        """
        Since the intrinsic-dimension calculation takes logarithm of the distances,
        if they are zero (or very small) it can cause numerical issues (NaN).
        """
        self.n_patches: int = n_patches
        self.k1: int = k1
        self.k2: int = k2
        self.k3: int = 4 * k2  # This will be used to plot a graph of the intrinsic-dimension per k (until k3)
        self.plot_graphs: bool = plot_graphs
        self.minimal_distance_between_patches = minimal_distance_between_patches

        # Sample a little bit more patches than requested, because later
        # we remove patches that are really close to one another
        # and we want our final number of patches to be the desired one.
        ratio_to_extend_n_patches = 1.5
        self.n_patches_extended = math.ceil(self.n_patches * ratio_to_extend_n_patches)

        # Will hold the kernel-sizes for each convolution-block,
        # and None for blocks which are not convolution-block.
        self.kernel_sizes: List[Optional[int]] = list()

    @torch.no_grad()
    def calc_int_dim_per_layer_all_stages(self, trainer, pl_module: pl.LightningModule):
        in_training_mode = pl_module.training
        pl_module.eval()  # will be a no-op if the model was not in training mode (i.e. in_training_mode = False)
        dataloaders_and_names = (
            (trainer.request_dataloader(RunningStage.TRAINING), 'train'),
            (trainer.request_dataloader(RunningStage.VALIDATING)[0], 'test'),
            (trainer.request_dataloader(RunningStage.VALIDATING)[1], 'test_no_aug')
        )
        for dataloader, name in dataloaders_and_names:
            self.calc_int_dim_per_layer_on_dataloader(trainer, pl_module, dataloader, name)
        pl_module.train(mode=in_training_mode)

    @staticmethod
    def plot_dim_per_k_graph(trainer, block_name, estimate_mean_over_data_points):
        x_axis_name: str = 'k'
        y_axis_name: str = f'{block_name} k-th int-dim'
        values = torch.stack([torch.arange(start=1, end=len(estimate_mean_over_data_points) + 1),
                              estimate_mean_over_data_points], dim=1)
        wandb_table = wandb.Table(columns=[x_axis_name, y_axis_name],
                                  data=values.cpu().tolist())
        line = wandb.plot.line(wandb_table, x_axis_name, y_axis_name,
                               title=f'{block_name} k-th int-dim per k')
        trainer.logger.experiment.log({f'{block_name}_ks_estimates': line})

    def calc_int_dim_per_layer_on_dataloader(self, trainer, pl_module, dataloader, name):
        """
        Given a VGG model, go over each block in it and calculates the intrinsic dimension of its input data.
        """
        for i in range(len(pl_module.features)):
            if self.kernel_sizes[i] is None:
                continue

            block_name = f'{name}_block_{i}'
            patches = self.get_patches_not_too_close_to_one_another(dataloader,  self.kernel_sizes[i],
                                                                    sub_model=pl_module.features[:i])

            estimates = get_estimates_matrix(patches, self.k3)
            estimate_mean_over_data_points = torch.mean(estimates, dim=0)

            if self.plot_graphs:
                IntrinsicDimensionCalculator.plot_dim_per_k_graph(trainer, block_name, estimate_mean_over_data_points)

            estimate_mean_over_k1_to_k2 = torch.mean(estimate_mean_over_data_points[self.k1:self.k2 + 1])
            extrinsic_dimension = patches.shape[1]
            intrinsic_dimension = estimate_mean_over_k1_to_k2.item()
            dimensions_ratio = intrinsic_dimension / extrinsic_dimension

            # TODO consider using `pl_module.log` to avoid wandb error "There's no data for the selected runs"
            trainer.logger.log_metrics({f'{block_name}_int_dim': intrinsic_dimension,
                                        f'{block_name}_ext_dim': extrinsic_dimension,
                                        f'{block_name}_dim_ratio': dimensions_ratio},
                                       step=trainer.global_step)

            pl_module.print(block_name)
            pl_module.print(f'\tIntrinsic-dimension = {intrinsic_dimension:.2f}')
            pl_module.print(f'\tExtrinsic-dimension = {extrinsic_dimension}')
            pl_module.print(f'\tRatio               = {dimensions_ratio:.4f}')

    def get_patches_not_too_close_to_one_another(self, dataloader, patch_size, sub_model):
        patches = self.get_flattened_patches(dataloader, patch_size, sub_model)
        patches_to_keep_mask = self.get_patches_to_keep_mask(patches)
        patches = patches[patches_to_keep_mask]
        # assert patches.shape[0] >= self.n_patches, \
        #     f"There are {patches.shape[0]} patches, which is less than requested ({self.n_patches}). "

        patches = patches[:self.n_patches]  # This is done to get exactly n_patches like the user requested.

        return patches

    def get_patches_to_keep_mask(self, patches):
        distance_matrix = torch.cdist(patches, patches)
        small_distances_indices = torch.nonzero(torch.less(distance_matrix, self.minimal_distance_between_patches))
        different_patches_mask = (small_distances_indices[:, 0] != small_distances_indices[:, 1])
        different_patches_close_indices_pairs = small_distances_indices[different_patches_mask]
        different_patches_close_indices = different_patches_close_indices_pairs.unique()
        patches_to_keep_mask = indices_to_mask(len(patches), different_patches_close_indices, negate=True)

        return patches_to_keep_mask

    def get_flattened_patches(self, dataloader, kernel_size, sub_model):
        patches = sample_random_patches(dataloader, self.n_patches_extended, kernel_size, sub_model)
        patches = patches.astype(np.float64)  # Increase accuracy of calculations.
        patches = torch.from_numpy(patches)
        patches = torch.flatten(patches, start_dim=1)

        return patches

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Calculates the intrinsic-dimension of the model, both on the training-data and test-data.
        """
        assert not pl_module.training, "Model should not be in training mode during validation"
        self.calc_int_dim_per_layer_all_stages(trainer, pl_module)

    def on_fit_start(self, trainer, pl_module):
        """
        Save the relevant information about the module (for example - kernel sizes of each layer).
        """
        for i in range(len(pl_module.features)):
            try:
                kernel_size_tuple = get_vgg_model_kernel_size(pl_module, block_index=i)
                assert kernel_size_tuple[0] == kernel_size_tuple[1], "Only square patches are supported"
                kernel_size = kernel_size_tuple[0]
                pl_module.print(f'Block {i} is a conv-block with kernel-size {kernel_size}. '
                                f'Input distribution intrinsic-dimension will be calculated. ')
            except ValueError:
                kernel_size = None
                pl_module.print(f'Block {i} is NOT a conv-block.               .'
                                f'Input distribution intrinsic-dimension will NOT be calculated. ')

            self.kernel_sizes.append(kernel_size)

        # # Calculate intrinsic dimension at initialization
        # self.calc_int_dim_per_layer_all_stages(trainer, pl_module)


def initialize(args: Args):
    model = LitVGG(args)
    datamodule = CIFAR10DataModule(args.data, args.opt.batch_size)
    wandb_logger = WandbLogger(project='thesis', config=args.flattened_dict())
    wandb_logger.watch(model)
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[IntrinsicDimensionCalculator(args.arch.n_patches,
                                                args.arch.k1, args.arch.k2,
                                                args.arch.plot_graphs)],
        max_epochs=args.opt.epochs,
        # limit_train_batches=5,  # TODO for debugging purposes
        # limit_val_batches=5,    # TODO for debugging purposes
        # limit_test_batches=5,   # TODO for debugging purposes
    )

    return model, trainer, datamodule


def main():
    args = get_args(args_class=Args)

    configure_logger(args.env.path)
    log_args(args)

    model, trainer, datamodule = initialize(args)
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
