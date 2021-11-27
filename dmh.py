import torch
import wandb
import torch.nn.functional as F
from loguru import logger
from patches import sample_random_patches
from schemas.dmh import Args
from utils import configure_logger, get_dataloaders, get_args, log_args, power_minus_1, train_model, get_model_device
from vgg import VGG, get_vgg_model_kernel_size


def initialize(args: Args):
    model = VGG(vgg_name=args.arch.model_name,
                final_mlp_n_hidden_layers=args.arch.final_mlp_n_hidden_layers,
                final_mlp_hidden_dim=args.arch.final_mlp_hidden_dim,
                dropout_prob=args.arch.dropout_prob,
                padding_mode=args.arch.padding_mode)
    model.to(args.env.device)

    wandb.init(project='thesis', config=args.flattened_dict())
    wandb.watch(model)

    optimizer = torch.optim.SGD(model.parameters(), args.opt.learning_rate, args.opt.momentum,
                                weight_decay=args.opt.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(get_model_device(model))
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

    return model, augmented_dataloaders, criterion, optimizer, scheduler


def get_estimates_matrix(data: torch.Tensor, k: int):
    """Calculates a matrix containing the intrinsic-dimension estimators.

    The ij-th cell contains the j-th estimate for the i-th data-point.
    In the notation of the paper below it's $\\hat(m)_j(x_i)$.

    See `Maximum Likelihood Estimation of Intrinsic Dimension
    <https://papers.nips.cc/paper/2004/file/74934548253bcab8490ebd74afed7031-Paper.pdf>`_
    """
    assert data.ndim == 2, f"data has shape {tuple(data.shape)}, expected (n, d) i.e. n d-dimensional vectors. "
    assert k <= data.shape[0], f"Number of data-points is {data.shape[0]} and k={k} should be smaller. "

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


def calc_intrinsic_dimension_per_layer(model,
                                       data_loader,
                                       n_patches,
                                       k1, k2,
                                       visualize: bool = False):
    """
    Given a VGG model, go over each block in it and calculates the intrinsic dimension of its input data.
    """
    for i in range(len(model.features)):
        try:
            kernel_size = get_vgg_model_kernel_size(model, i)
        except ValueError:
            logger.info(f'Skipping block {i} which is not a conv-block... ')
            continue
        assert kernel_size[0] == kernel_size[1], "Only square patches are supported"
        kernel_size = kernel_size[0]
        sub_model = model.features[:i]

        logger.info(f'Calculating intrinsic-dimension for block {i}...')
        patches = sample_random_patches(data_loader, n_patches, kernel_size, sub_model, visualize)
        patches_tensor = torch.from_numpy(patches)
        patches_flattened = torch.flatten(patches_tensor, start_dim=1)
        extrinsic_dimension = patches_flattened.shape[1]
        intrinsic_dimension = calc_intrinsic_dimension(patches_flattened, k1, k2)
        dimensions_ratio = intrinsic_dimension / extrinsic_dimension
        logger.info(f'Block {i}')
        logger.info(f'\tIntrinsic-dimension = {intrinsic_dimension:.2f}; ')
        logger.info(f'\tExtrinsic-dimension = {extrinsic_dimension}; ')
        logger.info(f'\tRatio               = {dimensions_ratio:.4f}')


def main():
    args = get_args(args_class=Args)

    configure_logger(args.env.path)
    log_args(args)

    clean_dataloaders = get_dataloaders(args.opt.batch_size,
                                        normalize_to_unit_gaussian=args.data.normalization_to_unit_gaussian,
                                        normalize_to_plus_minus_one=args.data.normalization_to_plus_minus_one)
    model, augmented_dataloaders, criterion, optimizer, scheduler = initialize(args)
    train_model(model, augmented_dataloaders, criterion, optimizer, scheduler, args.opt.epochs, args.env.log_interval)

    calc_intrinsic_dimension_per_layer(model, clean_dataloaders['train'],
                                       args.arch.n_patches, args.arch.k1, args.arch.k2)


if __name__ == '__main__':
    main()
