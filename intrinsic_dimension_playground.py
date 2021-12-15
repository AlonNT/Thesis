import torch
import wandb

import numpy as np
import pandas as pd
import plotly.express as px

from tqdm import tqdm
from scipy.stats import special_ortho_group

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


def get_intrinsic_dimension_estimates(data, start_k):
    estimates = get_estimates_matrix(torch.tensor(data), k=data.shape[0] - 1)
    estimate_mean_over_data_points = torch.mean(estimates, dim=0)
    estimate_mean_over_data_points = estimate_mean_over_data_points[start_k - 1:]
    return estimate_mean_over_data_points.numpy()


def gaussian_playground_main(args: Args):
    df = pd.DataFrame(0, index=np.arange(start=args.start_k, stop=args.n_points), columns=args.noise_std)
    df.index.name = 'k'
    df.columns.name = 'noise_level'

    figures = dict()
    for m, n in tqdm(list(zip(args.gaussian_dimension, args.extrinsic_dimension))):
        for noise_std in tqdm(args.noise_std):
            data = get_rotated_m_dimensional_gaussian_in_n_dimensional_space(m, n, args.n_points, noise_std)
            df[noise_std] = get_intrinsic_dimension_estimates(data, args.start_k)
            fig = px.line(df, title=f'Intrinsic dimension of {m}-dim gaussian in {n}-dim space')
            fig.add_hline(y=m, line_dash='dash')
            figures[f'int_dim_{m}_gauss_in_{n}_space'] = fig
    wandb.log(figures)


def main():
    args = get_args(args_class=Args)

    configure_logger(args.env.path)
    log_args(args)
    wandb.init(project='thesis', name='intrinsic_dimension_playground', config=args.flattened_dict())

    if args.gaussian_playground:
        gaussian_playground_main(args)


if __name__ == '__main__':
    main()
