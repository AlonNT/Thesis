import torch
import wandb
import faiss
import tikzplotlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

from tqdm import tqdm
from math import ceil
from loguru import logger
from scipy.stats import special_ortho_group
from sklearn.decomposition import PCA
from pathlib import Path

from patches import sample_random_patches
from dmh import (DataModule, 
                 get_estimates_matrix, 
                 get_patches_not_too_close_to_one_another, 
                 log_final_estimate,
                 LocallyLinearNetwork)
from schemas.intrinsic_dimension_playground import Args
from schemas.dmh import Args as ArgsForDMH
from utils import get_args, configure_logger, log_args, calc_whitening_from_dataloader, get_args_from_flattened_dict


def get_m_dimensional_gaussian_in_n_dimensional_space(m: int, n: int, n_points: int, noise_std: float = 0):
    mean = np.zeros(n)
    cov_diagonal = np.pad(np.ones(m), pad_width=(0, n - m), constant_values=noise_std)
    cov = np.diag(cov_diagonal)
    gaussian = np.random.default_rng().multivariate_normal(mean, cov, size=n_points)

    return gaussian


def get_rotated_m_dimensional_gaussian_in_n_dimensional_space(m: int, n: int, n_points: int, noise_std: float = 0):
    gaussian = get_m_dimensional_gaussian_in_n_dimensional_space(m, n, n_points, noise_std)

    random_rotation_generator = special_ortho_group(dim=n)
    random_rotation_matrix = random_rotation_generator.rvs()
    rotated_gaussian = np.dot(gaussian, random_rotation_matrix)

    if n == 3:
        wandb.log({f'gaussian_noise_{noise_std}': wandb.Object3D(rotated_gaussian)})

    return rotated_gaussian


def get_intrinsic_dimension_estimates(data, start_k, end_k=None):
    estimates = get_estimates_matrix(torch.tensor(data), k=data.shape[0] - 1)
    estimate_mean_over_data_points = torch.mean(estimates, dim=0)
    estimate_mean_over_data_points = estimate_mean_over_data_points[start_k - 1:end_k]
    return estimate_mean_over_data_points.numpy()


def get_pca_singular_values(data):
    pca = PCA()
    pca.fit(data)

    return pca.singular_values_


def get_figure_int_dim_different_dataset_sizes(n, m, noise_std, base_n_points, start_k, powers,
                                               base_factor=0.5, end_k=200):
    n_points_options = [ceil(base_n_points * (base_factor * 2 ** p)) for p in powers]
    df = pd.DataFrame(0, index=np.arange(start=start_k, stop=end_k + 1), columns=n_points_options)
    df.index.name = 'k'
    df.columns.name = 'n_points'
    for n_points in n_points_options:
        data = get_rotated_m_dimensional_gaussian_in_n_dimensional_space(m, n, n_points, noise_std)
        df[n_points] = get_intrinsic_dimension_estimates(data, start_k, end_k)
    fig = px.line(df)
    fig.add_hline(y=m, line_dash='dash')

    return fig


def gaussian_playground_main(args: Args):
    figures = dict()
    for m, n in tqdm(list(zip(args.gaussian_dimension, args.extrinsic_dimension))):
        df_mle = pd.DataFrame(0, index=np.arange(start=args.start_k, stop=args.n_points), columns=args.noise_std)
        df_mle.index.name = 'k'
        df_mle.columns.name = 'noise_level'

        df_pca = pd.DataFrame(0, index=np.arange(start=1, stop=n + 1), columns=args.noise_std)
        df_pca.index.name = 'i'
        df_pca.columns.name = 'noise_level'

        for noise_std in tqdm(args.noise_std):
            data = get_rotated_m_dimensional_gaussian_in_n_dimensional_space(m, n, args.n_points, noise_std)
            df_mle[noise_std] = get_intrinsic_dimension_estimates(data, args.start_k)
            df_pca[noise_std] = get_pca_singular_values(data)

        fig_mle = px.line(df_mle, title=f'MLE int-dim estimator of '
                                        f'{args.n_points} points sampled from '
                                        f'{m}-dim gaussian in {n}-dim space')
        fig_mle.add_hline(y=m, line_dash='dash')
        figures[f'int_dim_{m}_gauss_in_{n}_space'] = fig_mle

        fig_pca = px.line(df_pca, markers=True, title=f'Singular values of '
                                                      f'{args.n_points} points sampled from '
                                                      f'{m}-dim gaussian in {n}-dim space')
        figures[f'singular_values_{m}_gauss_in_{n}_space'] = fig_pca

    n = args.extrinsic_dimension[-1]
    m = args.gaussian_dimension[-1]
    k = f'singular_values_{m}_gauss_in_{n}_different_sizes'
    figures[k] = get_figure_int_dim_different_dataset_sizes(n, m, noise_std=0, base_n_points=args.n_points,
                                                            start_k=args.start_k, powers=range(3))

    wandb.log(figures)


def create_data_dict(patches: np.ndarray, whitening_matrix: np.ndarray,
                     normalize_patches_to_unit_vectors: bool = False):
    assert patches.ndim == 2, f'patches has shape {patches.shape}, it should have been flattened.'
    n_patches, d = patches.shape
    rng = np.random.default_rng()
    patches_shuffled = rng.permuted(patches, axis=1)
    patches_whitened = patches @ whitening_matrix
    patches_whitened_shuffled = rng.permuted(patches_whitened, axis=1)
    random_data = rng.multivariate_normal(mean=np.zeros(d), cov=np.eye(d), size=n_patches)

    data_dict = {
        'original': patches,
        'whitened': patches_whitened,
        'whitened_shuffled': patches_whitened_shuffled,
        'shuffled': patches_shuffled,
        'random': random_data
    }

    # faiss must receive float32 as input
    data_dict = {data_name: data.astype(np.float32) for data_name, data in data_dict.items()}

    if normalize_patches_to_unit_vectors:
        return normalize_data(data_dict)
    
    return data_dict


def normalize_data(data_dict):
    norms_dict = {data_name: np.linalg.norm(data, axis=1) 
                  for data_name, data in data_dict.items()}

    low_norms_masks_dict = {data_name: (norms < np.percentile(norms, q=0.1)) 
                            for data_name, norms in norms_dict.items()}

    filtered_data_dict = {data_name: data[np.logical_not(low_norms_masks_dict[data_name])] 
                          for data_name, data in data_dict.items()}

    filtered_norms_dict = {data_name: norms[np.logical_not(low_norms_masks_dict[data_name])] 
                           for data_name, norms in norms_dict.items()}

    normalized_data_dict = {data_name: (filtered_data_dict[data_name] / filtered_norms_dict[data_name][:, np.newaxis])
                            for data_name in data_dict.keys()}

    return normalized_data_dict


def cifar_mle_main(args: Args):
    datamodule = DataModule(args.data, batch_size=128)
    datamodule.prepare_data()
    datamodule.setup(stage='fit')
    dataloader = datamodule.train_dataloader_clean()
    n_points_list = args.int_dim.n_points
    patch_size_list = args.int_dim.patch_size
    if isinstance(n_points_list, int):
        n_points_list = [n_points_list]
    if isinstance(patch_size_list, int):
        patch_size_list = [patch_size_list]

    figures = dict()

    for patch_size in patch_size_list:
        logger.info(f'Calculating the whitening-matrix for patch-size {patch_size}x{patch_size}...')
        whitening_matrix = calc_whitening_from_dataloader(
            dataloader, patch_size, args.int_dim.whitening_regularization_factor, args.int_dim.zca_whitening)
        for n_points in n_points_list:
            prefix = f'{n_points:,}_{patch_size}x{patch_size}_patches'
            logger.info(f'Starting with {prefix}')

            patches = get_patches_not_too_close_to_one_another(dataloader, n_points, patch_size).numpy()

            if len(patches) < n_points:
                logger.warning(f'Number of patches {len(patches)} is lower than requested {n_points}')
        
            data_dict = create_data_dict(patches, whitening_matrix, args.int_dim.normalize_patches_to_unit_vectors)
            data_dict = {data_name: torch.from_numpy(data) for data_name, data in data_dict.items()}
            estimates_matrices = {data_name: get_estimates_matrix(data, 8 * args.int_dim.k2) 
                                  for data_name, data in data_dict.items()}
            estimates_dict = {data_name: torch.mean(estimates_matrix, dim=0) 
                               for data_name, estimates_matrix in estimates_matrices.items()}
            
            max_k = list(estimates_dict.values())[0].numel() + 1
            df = pd.DataFrame({data_name: estimates[args.int_dim.start_k - 1:] 
                               for data_name, estimates in estimates_dict.items()}, 
                              index=np.arange(args.int_dim.start_k, max_k))
            df.index.name = 'k'
            df.columns.name = 'type-of-data'
            figures[f'{prefix}-int_dim_per_k'] = px.line(df)

            # mle_int_dim, ratio = log_final_estimate(figures, estimates, patches.shape[1], prefix, args.int_dim.k1, args.int_dim.k2)
            # logger.info(f'{prefix}\tmle_int_dim {mle_int_dim:.2f} ({100 * ratio:.2f}% of ext_sim {patches.shape[1]})')
    
    wandb.log(figures, step=0)



def cifar_elbow_main(args: Args):
    datamodule = DataModule(args.data, batch_size=128)
    datamodule.prepare_data()
    datamodule.setup(stage='fit')
    dataloader = datamodule.train_dataloader_clean()

    n_points_list = args.int_dim.n_points
    patch_size_list = args.int_dim.patch_size
    if isinstance(n_points_list, int):
        n_points_list = [n_points_list]
    if isinstance(patch_size_list, int):
        patch_size_list = [patch_size_list]

    figures = dict()
    n_centroids_list = list(range(args.int_dim.min_n_centroids, args.int_dim.max_n_centroids + 1))

    for patch_size in patch_size_list:
        logger.info(f'Calculating the whitening-matrix for patch-size {patch_size}x{patch_size}...')
        whitening_matrix = calc_whitening_from_dataloader(
            dataloader, patch_size, args.int_dim.whitening_regularization_factor, args.int_dim.zca_whitening)
        for n_points in n_points_list:
            prefix = f'{n_points:,}_{patch_size}x{patch_size}_patches'
            logger.info(f'Starting with {prefix}')

            patches = sample_random_patches(dataloader, n_points, patch_size, verbose=True)
            patches_flat = patches.reshape(patches.shape[0], -1)
            n_patches, patch_dim = patches_flat.shape

            data_dict = create_data_dict(patches_flat, whitening_matrix)
            normalized_data_dict = normalize_data(data_dict)

            norms_dict = {data_name: np.linalg.norm(data, axis=1) 
                          for data_name, data in data_dict.items()}
            low_norms_masks_dict = {data_name: (norms < np.percentile(norms, q=0.1)) 
                                    for data_name, norms in norms_dict.items()}

            norms_df = pd.DataFrame(norms_dict)
            for col in norms_df.columns:
                figures[f'{prefix}_{col}_norms'] = px.histogram(norms_df, x=col, marginal='box')

            # {'mean','max','median'} 
            #     {'normalized-data', 'data'} 
            #         {'normalized-distance', 'distance'}
            #             {'original', 'shuffled', ...}
            #                 A list of values.
            dist_to_centroid = dict()  
            norm_data_names = ['normalized-data']  # ['normalized-data', 'data']
            norm_dist_names = ['distance']  # ['normalized-distance', 'distance']
            agg_funcs = {'mean': np.mean, 'max': np.max, 'median': np.median}
            for agg_name in agg_funcs.keys():
                dist_to_centroid[agg_name] = dict()
                for norm_data_name in norm_data_names:
                    dist_to_centroid[agg_name][norm_data_name] = dict()
                    for norm_dist_name in norm_dist_names:
                        dist_to_centroid[agg_name][norm_data_name][norm_dist_name] = dict()
                        for data_name in data_dict.keys():
                            dist_to_centroid[agg_name][norm_data_name][norm_dist_name][data_name] = list()

            for data_name in data_dict.keys():
                for norm_data_name in norm_data_names:
                    for k in tqdm(n_centroids_list, desc=f'Running k-means on {data_name} {norm_data_name} for different values of k'):
                        if (norm_data_name == 'normalized-data'):
                            data = normalized_data_dict[data_name]
                        else:
                            data = data_dict[data_name]
                            
                        kmeans = faiss.Kmeans(patch_dim, k)
                        kmeans.train(data)
                        distances, _ = kmeans.assign(data)

                        for norm_dist_name in norm_dist_names:
                            # If the data is already normalized, no need to divide by the norm.
                            if (norm_dist_name == 'normalized-distance') and (norm_data_name != 'normalized-data'):
                                not_low_norm_mask = np.logical_not(low_norms_masks_dict[data_name])
                                distances_filtered = distances[not_low_norm_mask]
                                norms_filtered = norms_dict[data_name][not_low_norm_mask]
                                distances = distances_filtered / norms_filtered

                            for agg_name, agg_func in agg_funcs.items():
                                dist_to_centroid[agg_name][norm_data_name][norm_dist_name][data_name].append(agg_func(distances))
            
            for agg_name in agg_funcs.keys():
                for norm_data_name in norm_data_names:
                    for norm_dist_name in norm_dist_names:
                        df = pd.DataFrame(data=dist_to_centroid[agg_name][norm_data_name][norm_dist_name], index=n_centroids_list)
                        df.name = f'{norm_data_name}-{agg_name}-{norm_dist_name}-distance-to-centroid'
                        df.index.name = 'k'
                        df.columns.name = 'type-of-data'
                        figures[f'{prefix}-{df.name}'] = px.line(df)

                        plt.figure()
                        plt.style.use("ggplot")
                        for col in df.columns:
                            plt.plot(n_centroids_list, df[col], label=col)
                        plt.legend()
                        plt.xlabel('k')
                        plt.ylabel('mean-distance')
                        plt.grid(True)
                        tikzplotlib.save(f'{df.name}.tex')
    
    wandb.log(figures, step=0)
 

def linear_regions_main(args: Args, wandb_run):

    artifact = wandb_run.use_artifact(args.int_dim.model_path, type='model')
    artifact_dir = artifact.download()
    checkpoint_path = str(Path(artifact_dir) / "model.ckpt")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    hparams = checkpoint['hyper_parameters']

    model_args: ArgsForDMH = get_args_from_flattened_dict(
        ArgsForDMH, hparams,
        excluded_categories=['env']  # Ignore environment args, such as GPU (which will raise error if on CPU).
    )
    model_args.env = args.env
    model = LocallyLinearNetwork.load_from_checkpoint(checkpoint_path, map_location=torch.device('cpu'), args=model_args)
    conv_block = model.embedding

    datamodule = DataModule(args.data, batch_size=128)
    datamodule.prepare_data()
    datamodule.setup(stage='validate')
    dataloader = datamodule.val_dataloader()

    figures = dict()

    patch_size = args.int_dim.patch_size
    n_points = args.int_dim.n_points
    n_centroids = args.int_dim.max_n_centroids

    logger.info(f'Calculating the whitening-matrix for patch-size {patch_size}x{patch_size}...')
    patches = sample_random_patches(dataloader, n_points, patch_size, verbose=True)
    patches_flat = patches.reshape(patches.shape[0], -1)
    n_patches, patch_dim = patches_flat.shape

    kmeans = faiss.Kmeans(patch_dim, n_centroids, verbose=True)
    kmeans.train(patches_flat)
    centroids = kmeans.centroids.reshape((n_centroids,) + patches.shape[1:])
    _, indices = kmeans.assign(patches_flat)

    patches_activations = None
    centroids_activations = None

    with torch.no_grad():
        patches_activations = conv_block(torch.from_numpy(patches)).detach().cpu().numpy().squeeze()
        centroids_activations = conv_block(torch.from_numpy(centroids)).detach().cpu().numpy().squeeze()

    centroids_of_patches_activations = centroids_activations[indices]

    patches_active_neurons = (patches_activations > 0)
    centroids_active_neurons = (centroids_of_patches_activations > 0)
    different_activations = (patches_active_neurons != centroids_active_neurons)
    fraction_different_activations = np.mean(different_activations, axis=1)
    
    agg_funcs = {'mean': np.mean, 'max': np.max, 'median': np.median}
    for agg_name, agg_func in agg_funcs.items():
        logger.info(f'{agg_name}_symmetrical_difference = {agg_func(fraction_different_activations)}')

    figures[f'activation_set_symmetrical_difference'] = px.histogram(fraction_different_activations, marginal='box')

    wandb.log(figures, step=0)


def main():
    args = get_args(args_class=Args)

    configure_logger(args.env.path)
    log_args(args)
    wandb_run = wandb.init(project='thesis', name=args.env.wandb_run_name, config=args.flattened_dict())

    if args.int_dim.gaussian_playground:
        gaussian_playground_main(args)
    if args.int_dim.cifar_mle:
        cifar_mle_main(args)
    if args.int_dim.linear_regions:
        linear_regions_main(args, wandb_run)
    if args.int_dim.cifar_elbow:
        cifar_elbow_main(args)
    

if __name__ == '__main__':
    main()
