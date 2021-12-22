import torch
import wandb

import numpy as np
import pandas as pd
import plotly.express as px

from tqdm import tqdm
from math import ceil
from scipy.stats import special_ortho_group
from sklearn.decomposition import PCA

from dmh import get_estimates_matrix
from schemas.intrinsic_dimension_playground import Args
from utils import get_args, configure_logger, log_args


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
                                                            start_k=args.start_k, powers=range(5))

    wandb.log(figures)


def main():
    args = get_args(args_class=Args)

    configure_logger(args.env.path)
    log_args(args)
    wandb.init(project='thesis', name='intrinsic_dimension_playground', config=args.flattened_dict())

    gaussian_playground_main(args)


if __name__ == '__main__':
    main()
