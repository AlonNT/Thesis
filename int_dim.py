import copy
import math
import sys
import torch

import numpy as np
import pandas as pd
import plotly.express as px
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger
from typing import Optional, List, Tuple, Dict
from tqdm import tqdm

from pytorch_lightning import Callback
from pytorch_lightning.trainer.states import RunningStage

from patches import sample_random_patches
from schemas.dmh import Args, DMHArgs
from utils import (configure_logger, get_args, get_model_device, power_minus_1, whiten_data, normalize_data,
                   calc_whitening_from_dataloader, run_kmeans_clustering, DataModule, initialize_datamodule,
                   initialize_wandb_logger, LitVGG, initialize_model, initialize_trainer, unwatch_model)


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
    def __init__(self,
                 patches: np.ndarray,
                 values: np.ndarray,
                 stride: int,
                 padding: int,
                 k: int,
                 use_faiss: bool = False,
                 use_linear_function: str = 'none',
                 use_angles: bool = False,
                 whitening_matrix: Optional[np.ndarray] = None):
        super(NeighborsValuesAssigner, self).__init__()
        self.kernel_size = patches.shape[-1]
        self.stride = stride
        self.padding = padding
        self.k = k
        self.use_faiss = use_faiss
        self.use_linear_function = use_linear_function
        self.use_angles = use_angles
        self.whitening_matrix = whitening_matrix

        patches_flat = patches.reshape(patches.shape[0], -1)
        if self.whitening_matrix is not None:
            patches_flat = patches_flat @ self.whitening_matrix

        if self.use_angles:
            patches_flat /= (np.linalg.norm(patches_flat, axis=1)[..., np.newaxis] + 0.001)

        if self.use_faiss:
            raise NotImplementedError(f'`faiss` is not implemented yet in this context. '
                                      f'Needs to wrap `index` in a function, like `run_kmeans_clustering`.')
            self.index = faiss.IndexFlatL2(patches_flat.shape[1])
            self.index.add(patches_flat)
        else:
            bias = 0.5 * (np.linalg.norm(patches_flat, axis=1) ** 2)
            if self.whitening_matrix is not None:
                patches_flat = patches_flat @ self.whitening_matrix.T
                patches = patches_flat.reshape(-1, *patches.shape[1:])
            kernel = -1 * patches

            self.knn_indices_calculator = KNearestPatchesEmbedding(kernel, bias, stride, padding, k,
                                                                   return_as_mask=False)
        if values.ndim > 2:
            assert use_linear_function == 'none', 'the argument use_linear_function is for values assigner mode.'
            assert not use_faiss, 'Can not use faiss when using linear function per neighbor mode.'
            assert values.ndim == 3, f'values has {values.ndim} dimensions, should be 2 or 3.'
        values_requires_grad = (values.ndim == 3)
        self.values = nn.Parameter(torch.Tensor(values), requires_grad=values_requires_grad)

        if use_linear_function == 'full':
            values_dim = self.values.shape[1]
            self.conv = nn.Conv2d(in_channels=k*values_dim, out_channels=values_dim, kernel_size=(1, 1))
        elif use_linear_function == 'partial':
            self.conv = nn.Conv2d(in_channels=k, out_channels=1, kernel_size=(1, 1))
        else:  # use_linear_function == 'none'
            self.conv = None

    def reduce_across_neighbors(self, result: torch.Tensor):
        batch_size, k, values_dim, height, width = result.shape
        assert k == self.k, f'Dimension 1 in argument `result` should equal {self.k=} but it is {k}'
        if self.use_linear_function == 'full':
            result = torch.flatten(result, start_dim=1, end_dim=2)
            result = self.conv(result)
        elif self.use_linear_function == 'partial':
            result = torch.swapaxes(result, 1, 2)
            result = torch.flatten(result, start_dim=0, end_dim=1)
            result = self.conv(result)
            result = torch.reshape(result, shape=(batch_size, values_dim, height, width))
        else:  # self.use_linear_function == 'none'
            result = torch.mean(result, dim=1)

        return result

    def neighbors_linear_functions(self, x: torch.Tensor) -> torch.Tensor:
        values_dim, patch_dim = self.values.shape[1:]
        knn_indices = self.knn_indices_calculator(x)
        batch_size, k, output_height, output_width = knn_indices.shape
        assert k == self.k, f'self.knn_indices_calculator did not return {self.k} indices'

        x_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        batch_size, patch_dim, n_patches = x_unfold.shape
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.flatten(start_dim=0, end_dim=1)
        x_unfold = x_unfold.unsqueeze(-1)  # Now (batch_size * n_patches, patch_dim, 1)

        matrices = self.values[knn_indices]
        matrices = matrices.permute(0, 2, 3, 1, 4, 5)
        matrices = matrices.flatten(start_dim=0, end_dim=2)
        matrices = matrices.flatten(start_dim=1, end_dim=2)  # Now (batch_size * n_patches, k * values_dim, patch_dim)

        result = torch.bmm(matrices, x_unfold)
        result = result.reshape(batch_size, output_height, output_width, self.k, values_dim)
        result = result.permute(0, 3, 4, 1, 2)  # Now (batch_size, k, values_dim, output_height, output_width)

        result = self.reduce_across_neighbors(result)

        return result

    def neighbors_values_assigner(self, x: torch.Tensor) -> torch.Tensor:
        knn_indices = self.knn_indices_calculator(x)
        batch_size, k, output_height, output_width = knn_indices.shape
        assert k == self.k, f'self.knn_indices_calculator did not return {self.k} indices'

        result = self.values[knn_indices]
        result = torch.permute(result, dims=(0, 1, 4, 2, 3))

        result = self.reduce_across_neighbors(result)

        return result

    def forward_using_torch(self, x: torch.Tensor) -> torch.Tensor:
        if self.values.ndim == 2:
            return self.neighbors_values_assigner(x)
        else:  # self.values.ndim == 3
            return self.neighbors_linear_functions(x)

    def forward_using_faiss(self, x: torch.Tensor) -> torch.Tensor:
        x_unfolded = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        batch_size, patch_dim, n_patches = x_unfolded.shape
        values_channels = self.values.shape[1]
        output_spatial_size = int(math.sqrt(n_patches))

        # Transpose from (N, C*H*W, M) to (N, M, C*H*W) and then reshape to (N*M, C*H*W) to have collection of vectors
        # Also make contiguous in memory (required by function kmeans.search).
        x_unfolded = x_unfolded.transpose(dim0=1, dim1=2).flatten(start_dim=0, end_dim=1).contiguous().cpu().numpy()
        if self.whitening_matrix is not None:
            x_unfolded = x_unfolded @ self.whitening_matrix
        _, indices = self.index.search(x_unfolded, self.k)
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


def get_estimates_matrix(data: torch.Tensor, k: int):
    """Calculates a matrix containing the intrinsic-dimension estimators.

    See `Maximum Likelihood Estimation of Intrinsic Dimension
    <https://papers.nips.cc/paper/2004/file/74934548253bcab8490ebd74afed7031-Paper.pdf>`_

    Args:
        data: The data of shape (n, d) i.e. n d-dimensional vectors.
        k: The number of neighbors which define the size of the estimates-matrix.

    Returns:
        A matrix of shape (n, k) where the ij-th cell contains the j-th estimate for the i-th data-point.
        In the notation of the paper it's $\\hat(m)_j(x_i)$.
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

    Args:
        data: The data of shape (n, d) i.e. n d-dimensional vectors.
        k1: The nearest-neighbor index to begin the mean with.
        k1: The nearest-neighbor index to end the mean with.

    Returns:
        The intrinsic-dimension of the given data.
    """
    estimates = get_estimates_matrix(data, k2)
    estimate_mean_over_data_points = torch.mean(estimates, dim=0)
    estimate_mean_over_k1_to_k2 = torch.mean(estimate_mean_over_data_points[k1:k2 + 1])

    return estimate_mean_over_k1_to_k2.item()


def indices_to_mask(n, indices, negate=False):
    mask = torch.zeros(n, dtype=torch.bool, device=indices.device).scatter_(dim=0, index=indices, value=1)
    if negate:
        mask = torch.bitwise_not(mask)
    return mask


def get_flattened_patches(dataloader, n_patches, kernel_size,
                          shuffle_before_estimate: bool = False, sub_model=None, device=None):
    """Sample patches from the given data and flatten them, possibly shuffling their content.

    Args:
        dataloader: The given dataloader to sample from.
        n_patches: The number of patches to sample.
        kernel_size: The size of the patches to sample (-1 means the whole image).
        shuffle_before_estimate: Whether to shuffle the pixels in each patch
            (used as a "baseline" for the intrinsic-dimension estimation)
        sub_model: The sub-model to run on the images generated from the dataloader
            (used when sampling patches in deeper layer and not in the first one)
        device: If given, move the patches to this device before returning them.

    Returns:
        A 2-dimensional tensor containing `n_patches` flattened patches.
    """
    patches = sample_random_patches(dataloader, n_patches, kernel_size, sub_model)
    patches = patches.reshape(patches.shape[0], -1)  # a.k.a. flatten in NumPy

    if shuffle_before_estimate:
        # Use a fresh new sampled permutation for each patch (i.e. row in the matrix).
        # This functionality does not exist in PyTorch and that's why it's being done in NumPy.
        patches = np.random.default_rng().permuted(patches, axis=1, out=patches)

    patches = patches.astype(np.float64)  # Increase accuracy of calculations later.
    patches = torch.from_numpy(patches)

    if device is not None:
        patches = patches.to(device)

    return patches


def get_patches_to_keep_mask(patches, minimal_distance: float = 1e-05):
    """Calculates a mask indicating the patches to keep (distant from one another) from the given patches tensor.

    Args:
        patches: The given patches to analyze.
        minimal_distance: The minimal distance to allow between different patches.

    Returns:
        A boolean mask (as a tensor) indicating patches to keep (distant at least `minimal_distance` from one another).
    """
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
                                             sub_model=None, device=None) -> torch.Tensor:
    """Sample patches from the given data, such that they are at least `minimal_distance` apart from one another.

    Since the intrinsic-dimension calculation takes logarithm of the distances,
    if they are zero (or very small) it can cause numerical issues (NaN).
    The solution is to sample a bit more patches than requested,
    and later we remove patches that are really close to one another,
    and we want our final number of patches to be the desired one.
    Since there is a lower probability to get similar images rather than patches,
    the exact number is different.

    Args:
        dataloader: The dataloader to sample patches from.
        n_patches: Number of patches to sample.
        patch_size: The side of the patches to sample (-1 means the whole image).
        minimal_distance: The minimal distance to allow between adjacent patches.
        shuffle_before_estimate: Whether to shuffle the pixels in each patch
            (used as a "baseline" for the intrinsic-dimension estimation)
        sub_model: The sub-model to run on the images generated from the dataloader
            (used when sampling patches in deeper layer and not in the first one)
        device: The device to run the forward-pass of the sub-model.

    Returns:
        A tensor containing the sampled patches (will be exactly `n_patches` as requested,
        but it might be a bit less if for some reason a lot of sampled patches were too
        close to each other).
    """
    ratio_to_extend_n = 1.5
    n_patches_extended = math.ceil(n_patches * ratio_to_extend_n)
    patches = get_flattened_patches(dataloader, n_patches_extended, patch_size, 
                                    shuffle_before_estimate, sub_model, device)
    patches_to_keep_mask = get_patches_to_keep_mask(patches, minimal_distance)
    patches = patches[patches_to_keep_mask]

    patches = patches[:n_patches]  # This is done to get exactly (or up-to) n like the user requested

    return patches


def log_dim_per_k_graph(estimates: np.ndarray):
    """Creates a graph of the k-th intrinsic dimension per k and adds it to the given `metrics` dictionary

    Args:
        estimates: A NumPy vector containing the k-th intrinsic-dimension estimates for k=1,2,...,K
            (where K is the maximal k chosen to calculate the intrinsic-dimension).

    Returns:
        A plotly-express line object.
    """
    min_k = 5
    max_k = len(estimates) + 1
    df = pd.DataFrame(estimates[min_k - 1:], index=np.arange(min_k, max_k), columns=['k-th intrinsic-dimension'])
    df.index.name = 'k'
    return px.line(df)


def log_singular_values(metrics: dict, prefix: str, data: np.ndarray):
    """Create several graphs analyzing the data singular values (and adding them to the given `metrics` dictionary).

    The graphs that are being added to the `metrics` dictionary (with the prefix f'{prefix}-') are:
    - d_cov: "The covariance dimension" which is the smallest number of dimensions
        needed to explain 95% of the total variance.
        Inspiration (and the name 'd_cov') taken from
        "The Unreasonable Effectiveness of Patches in Deep Convolutional Kernels Methods"
    - singular_values: The singular values of the data (i.e. X = USV^T and it's the values in the diagonal matrix S).
    - singular_values_ratio: The singular values, normalized by the largest one (i.e. \lambda_1).
        Inspiration (and the name 'd_cov') taken from
        "The Unreasonable Effectiveness of Patches in Deep Convolutional Kernels Methods"
    - variance_ratio: The sum of the singular values until i, divided by the total sum of the singular values.
    - explained_variance_ratio: The squared singular values divided by the number of data-points, divided by
        the sum of all of these values.
        Inspiration taken from sklearn.decomposition._pca.PCA._fit_full
    - reconstruction_errors: The error (l2 distance) between the original data-points and the reconstruced ones.
        The reconstruction is being done by transforming to R^k (for k=1,2,...,min{100, patches-dimension})
        using the first k singular vectors, then transforming it back to the original dimension.
    - normalized_reconstruction_errors: The reconstruction error, now normalized by the norm of the data-point.

    Args:
        metrics: The dictionary containing the different metrics that will be logged to wandb.
            The different plots will be added to this dictionary.
        prefix: The prefix of the keys that the generated plots will be inserted to the `metrics` dictionary with.
        data: The data as a NumPy array of shape (n_samples, n_features), i.e. a collection of row-vectors.
    """
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
    normalized_reconstruction_errors = dict()
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

        explained_variance = (s ** 2) / (n - 1)
        explained_variance_ratio[data_name] = explained_variance / explained_variance.sum()

        variance_ratio[data_name] = np.cumsum(s) / np.sum(s)
        singular_values_ratio[data_name] = s / s[0]

        logger.debug('Calculating reconstruction error...')
        transformed_data = np.dot(data_orig, v)  # n x d matrix (like the original)
        reconstruction_errors_list = list()
        normalized_reconstruction_errors_list = list()
        for k in range(1, min(d+1, 101)):
            v_reduced = v[:, :k]  # d x k matrix
            transformed_data_reduced = np.dot(transformed_data, v_reduced)  # n x k matrix
            transformed_data_reconstructed = np.dot(transformed_data_reduced, v_reduced.T)  # n x d matrix
            data_reconstructed = np.dot(transformed_data_reconstructed, v_t)  # n x d matrix
            error_per_data_point = np.linalg.norm(data_orig - data_reconstructed, axis=1)
            data_points_norms = np.linalg.norm(data_orig, axis=1)
            normalized_error_per_data_point = error_per_data_point / data_points_norms
            reconstruction_error = np.mean(error_per_data_point)
            normalized_reconstruction_error = np.mean(normalized_error_per_data_point)
            reconstruction_errors_list.append(reconstruction_error)
            normalized_reconstruction_errors_list.append(normalized_reconstruction_error)
        logger.debug('Finished calculating reconstruction error.')

        reconstruction_errors[data_name] = np.array(reconstruction_errors_list)
        normalized_reconstruction_errors[data_name] = np.array(normalized_reconstruction_errors_list)

    metrics[f'{prefix}-d_cov'] = np.where(variance_ratio['original_data'] > 0.95)[0][0]

    fig_args = dict(markers=True)
    metrics[f'{prefix}-singular_values'] = px.line(pd.DataFrame(singular_values), **fig_args)
    metrics[f'{prefix}-singular_values_ratio'] = px.line(pd.DataFrame(singular_values_ratio), **fig_args)
    metrics[f'{prefix}-variance_ratio'] = px.line(pd.DataFrame(variance_ratio), **fig_args)
    metrics[f'{prefix}-explained_variance_ratio'] = px.line(pd.DataFrame(explained_variance_ratio), **fig_args)
    metrics[f'{prefix}-reconstruction_errors'] = px.line(pd.DataFrame(reconstruction_errors), **fig_args)
    metrics[f'{prefix}-normalized_reconstruction_errors'] = px.line(pd.DataFrame(normalized_reconstruction_errors),
                                                                    **fig_args)


def log_final_estimate(metrics: dict,
                       estimates: torch.Tensor,
                       extrinsic_dimension: int,
                       block_name: str,
                       k1: int,
                       k2: int):
    """Logs the final MLE estimator for the intrinsic dimension (mean of the k-th estimates from `k1` to `k2`),
    both as the original number and as a fraction of the extrinsic-dimension.

    Args:
        metrics: The dictionary containing the different metrics that will be logged to wandb.
            returned values from the function will also be added to this dictionary.
        estimates: A NumPy vector containing the k-th intrinsic-dimension estimates for k=1,2,...,K
            (where K is the maximal k chosen to calculate the intrinsic-dimension).
        extrinsic_dimension: The extrinsic-dimension of the patches.
        block_name: The name of the block, will be the prefix of the keys that will be added to `metrics` dictionary.
        k1: The minimal k to take the mean to get the final MLP estimate.
        k2: The maximal k (inclusive) to take the mean to get the final MLP estimate.

    Returns:
        Two numbers - the intrinsic-dimension, and the ratio between it and the extrinsic-dimension.
    """
    estimate_mean_over_k1_to_k2 = torch.mean(estimates[k1:k2 + 1])
    intrinsic_dimension = estimate_mean_over_k1_to_k2.item()
    dimensions_ratio = intrinsic_dimension / extrinsic_dimension

    block_name = f'{block_name}-ext_dim_{extrinsic_dimension}'
    metrics.update({f'{block_name}-int_dim': intrinsic_dimension,
                    f'{block_name}-dim_ratio': dimensions_ratio})
    return intrinsic_dimension, dimensions_ratio


class IntrinsicDimensionCalculator(Callback):

    def __init__(self, args: DMHArgs):
        """Initialize a new instance of the intrinsic-dimension-calculator callback.

        This callback calculates the intrinsic-dimension of the input distribution to each layer in the model.
        It runs after each epoch (on_validation_epoch_end) and in the beginning and the end of the whole
        training process (on_fit_begin and on_fit_end) where it also does more heavy compute to generate graphs.

        Args:
            args: The arguments schema
        """
        self.args: DMHArgs = args

    def calc_int_dim_per_layer_on_dataloader(self, trainer, pl_module, dataloader, log_graphs: bool = False):
        """The main function of the callback - iterate the model's layers and estimate the intrinsic dimension
        of their input data distribution.

        Args:
            trainer: The trainer
            pl_module: The model (expected to be LitVGG or LitMLP)
            dataloader: The dataloader to sample data-points from.
            log_graphs: Whether to log graphs or not (heavy compute, that's why it happens only in
                beginning or in the end of the whole training process).
        """
        metrics = dict()
        for i in range(pl_module.num_blocks):
            block_name = f'block_{i}'
            estimate_dim_on_whole_image = self.args.estimate_dim_on_images or (i >= len(pl_module.kernel_sizes))
            patch_size = -1 if estimate_dim_on_whole_image else pl_module.kernel_sizes[i]
            patches = get_patches_not_too_close_to_one_another(dataloader, self.args.n_clusters,
                                                               patch_size, self.args.minimal_distance,
                                                               self.args.shuffle_before_estimate,
                                                               pl_module.get_sub_model(i))

            estimates_matrix = get_estimates_matrix(patches, 8*self.args.k2 if log_graphs else self.args.k2)
            estimates = torch.mean(estimates_matrix, dim=0)

            if log_graphs:
                metrics[f'{block_name}-int_dim_per_k'] = log_dim_per_k_graph(estimates.cpu().numpy())
                log_singular_values(metrics, block_name, patches.cpu().numpy())

            mle_int_dim, ratio = log_final_estimate(metrics, estimates, patches.shape[1], block_name,
                                                    self.args.k1, self.args.k2)
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
                                                  dataloader=trainer.request_dataloader(RunningStage.VALIDATING),
                                                  log_graphs=log_graphs)
        pl_module.train(training_mode)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_step == 0:  # This happens in the end of validation loop sanity-check before training,
            return                    # and we do not want to treat it the same as actual validation-epoch end.
        self.calc_int_dim_per_layer(trainer, pl_module)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Since log_graphs=True takes a lot of time, we do it only in the beginning / end of the training process.
        """
        self.calc_int_dim_per_layer(trainer, pl_module, log_graphs=True)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Since log_graphs=True takes a lot of time, we do it only in the beginning / end of the training process.
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

        centroids, indices, distances = run_kmeans_clustering(patches_flat, self.args.n_clusters, self.args.use_faiss)
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
        trainer.logger.experiment.log(metrics, step=trainer.global_step, commit=True)

    def on_validation_epoch_end(self, trainer, pl_module: LitVGG):
        self.log_linear_regions_respect_per_layer(trainer, pl_module,
                                                  dataloader=trainer.request_dataloader(RunningStage.VALIDATING)[1])


class ImitatorKNN(pl.LightningModule):
    def __init__(self, teacher: LitVGG, args: Args, datamodule: DataModule):
        super().__init__()

        self.args: Args = args
        self.datamodule = datamodule

        self.imitators: nn.Sequential = nn.Sequential()
        self.features: nn.Sequential = copy.deepcopy(teacher.features)
        self.flatten: nn.Flatten = nn.Flatten()
        self.mlp: nn.Sequential = copy.deepcopy(teacher.mlp)

        # This is the teacher's block that will be used to train the linear functions imitating the removed block.
        self.teacher_block: Optional[nn.Sequential] = None

        self.imitated_blocks_output_channels: List[int] = list()
        self.imitated_blocks_conv_kwargs: List[Dict[str, int]] = list()

        self.loss = nn.CrossEntropyLoss()
        self.imitator_loss = nn.L1Loss()

    def get_dataloader_for_patches_sampling(self):
        # TODO Consider whitening / normalizing the patches.
        if self.args.dmh.dataset_type_for_patches_dictionary == 'aug':
            dataloader = self.datamodule.train_dataloader()
        elif self.args.dmh.dataset_type_for_patches_dictionary == 'no_aug':
            dataloader = self.datamodule.train_dataloader_no_aug()
        else:  # self.args.dmh.dataset_type_for_patches_dictionary == 'clean'
            dataloader = self.datamodule.train_dataloader_clean()

        return dataloader

    @torch.no_grad()
    def imitate_first_block(self):
        assert len(self.features) > 0, "This function should be called when there is some block to imitate."
        self.teacher_block = self.features[0]
        self.features = self.features[1:]

        self.teacher_block.requires_grad_(False)
        self.teacher_block.eval()

        teacher_conv: nn.Conv2d = self.teacher_block[0]

        self.imitated_blocks_output_channels.append(teacher_conv.out_channels)
        self.imitated_blocks_conv_kwargs.append({k: getattr(teacher_conv, k)
                                                 for k in ['kernel_size', 'dilation', 'padding', 'stride']})

        dataloader = self.get_dataloader_for_patches_sampling()

        patches = sample_random_patches(dataloader, self.args.dmh.n_patches,
                                        teacher_conv.kernel_size[0], self.imitators,
                                        random_uniform_patches=self.args.dmh.random_uniform_patches,
                                        random_gaussian_patches=self.args.dmh.random_gaussian_patches,
                                        verbose=True)
        patches_shape = patches.shape
        patches = patches.reshape(patches.shape[0], -1)

        whitening_matrix = None

        if self.args.dmh.use_whitening:
            # Calculates the whitening matrix and multiply each patch by this matrix,
            # so the kmeans later will run on the whitened patches.
            whitening_matrix = calc_whitening_from_dataloader(
                dataloader,
                patch_size=teacher_conv.kernel_size[0],
                whitening_regularization_factor=self.args.dmh.whitening_regularization_factor,
                zca_whitening=self.args.dmh.zca_whitening,
                existing_model=self.imitators
            )
            patches = patches @ whitening_matrix

        if self.args.dmh.n_patches > self.args.dmh.n_clusters:
            centroids, indices, distances = run_kmeans_clustering(patches, self.args.dmh.n_clusters,
                                                                  self.args.dmh.use_faiss)
            patches = centroids

        if self.args.dmh.use_whitening:
            # We want to run the teacher block on the original patches,
            # so the whitening will be used only for the distance calculation.
            patches = patches @ np.linalg.inv(whitening_matrix)

        patches = patches.reshape(-1, *patches_shape[1:])

        if self.args.dmh.imitate_with_knn:
            patches_tensor = torch.from_numpy(patches).to(self.device)
            patches_outputs = self.teacher_block(patches_tensor)
            patches_outputs = patches_outputs.cpu().numpy().squeeze(axis=(2, 3))
            values = patches_outputs
        elif self.args.dmh.imitate_with_locally_linear_model:
            patches_dict_size, patches_channels, patches_height, patches_width = patches.shape
            patches_dim = patches_channels * patches_height * patches_width
            out_channels = self.teacher_block[0].out_channels
            values = torch.empty(patches_dict_size, out_channels, patches_dim,
                                 requires_grad=True, device=self.device, dtype=torch.float32)

            # In order to define each linear layer with proper weights initialization (correct fan-in and fan-out)
            # it's being done in a for-loop using PyTorch default linear-layer initialization.
            for i in range(patches_dict_size):
                tmp_linear = torch.nn.Linear(in_features=patches_dim, out_features=out_channels,
                                             bias=False,  # TODO consider using bias=True
                                             device=self.device)
                values[i] = tmp_linear.weight.data
        else:
            raise ValueError('One of imitate_with_knn or imitate_with_locally_linear_model must be true.')

        imitator = NeighborsValuesAssigner(patches, values=values, stride=teacher_conv.stride[0],
                                           padding=teacher_conv.padding[0], k=self.args.dmh.k,
                                           use_faiss=self.args.dmh.use_faiss,
                                           use_linear_function=self.args.dmh.use_linear_function,
                                           use_angles=self.args.dmh.use_angles, whitening_matrix=whitening_matrix)

        loss = 0
        # If the mode is imitating with k-nearest-neighbors, evaluate the performance.
        # When imitating with locally linear functions, it needs to be trained because it's initialized randomly.
        if self.args.dmh.imitate_with_knn:
            for inputs, _ in tqdm(self.datamodule.val_dataloader(), desc=f'Evaluating imitated block'):
                inputs = self.imitators(inputs)
                batch_targets = self.teacher_block(inputs)
                batch_imitated_targets = imitator(inputs)
                loss += F.l1_loss(batch_targets, batch_imitated_targets)
            loss /= len(self.datamodule.val_dataloader())

        self.imitators = nn.Sequential(*self.imitators, imitator)

        # self.teacher_block is needed only for mode = imitate_with_locally_linear_model
        if not self.args.dmh.train_linear_functions_by_imitation:
            self.teacher_block = None

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
        predictions = torch.argmax(logits, dim=1)
        accuracy = torch.sum(labels == predictions).item() / len(labels)

        if (self.teacher_block is not None) and self.args.dmh.train_linear_functions_by_imitation:
            # This means the training-mode is to train the linear functions imitating the teacher_block.
            last_imitator_input = self.imitators[:-1](x)
            imitator_output = self.imitators[-1](last_imitator_input)
            teacher_block_output = self.teacher_block(last_imitator_input)
            loss = self.imitator_loss(imitator_output, teacher_block_output)
        else:
            loss = self.loss(logits, labels)

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
        # When training the linear functions to imitate the teacher block, the gradients are tiny.
        # That's why we increase the learning-rate dramatically for this phase of the training.
        learning_rate = self.args.opt.learning_rate
        if (self.teacher_block is not None) and self.args.dmh.train_linear_functions_by_imitation:
            learning_rate *= 10000

        optimizer = torch.optim.SGD(self.parameters(),
                                    learning_rate,
                                    self.args.opt.momentum,
                                    weight_decay=self.args.opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.args.opt.learning_rate_decay_steps,
                                                         gamma=self.args.opt.learning_rate_decay_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


def initialize_callbacks(args: DMHArgs):
    callbacks = list()
    if args.estimate_intrinsic_dimension:
        callbacks.append(IntrinsicDimensionCalculator(args))
    if args.linear_regions_calculator:
        callbacks.append(LinearRegionsCalculator(args))
    return callbacks


def main():
    args = get_args(args_class=Args)
    datamodule = initialize_datamodule(args.data, args.opt.batch_size)
    wandb_logger = initialize_wandb_logger(args)
    configure_logger(args.env.path, print_sink=sys.stdout, level='DEBUG')  # if args.env.debug else 'INFO')
    model = initialize_model(args.arch, args.opt, args.data, wandb_logger)
    callbacks = initialize_callbacks(args)

    if not args.arch.use_pretrained:
        wandb_logger.watch(model, log='all')
        trainer = initialize_trainer(args.env, args.opt, wandb_logger, callbacks)
        trainer.fit(model, datamodule=datamodule)
        unwatch_model(model)

    if args.dmh.imitate_with_knn or args.dmh.imitate_with_locally_linear_model:
        datamodule_for_sampling = initialize_datamodule(args.data, batch_size=4)
        imitator = ImitatorKNN(model, args, datamodule_for_sampling)
        for i in range(len(imitator.features)):
            replaced_block_error = imitator.imitate_first_block()

            if args.dmh.train_linear_functions_by_imitation:
                assert imitator.teacher_block is not None
                wandb_logger = initialize_wandb_logger(args, name_suffix=f'_block_{i}_imitator')
                wandb_logger.watch(imitator, log='all')
                trainer = initialize_trainer(args.env, args.opt, wandb_logger, callbacks)
                trainer.fit(imitator, datamodule=datamodule)
                imitator.teacher_block = None  # This will cause fine-tuning the `features` modules in the next `fit`
                imitator.imitators[-1].requires_grad_(False)
                imitator.imitators[-1].eval()
                unwatch_model(imitator)

            wandb_logger = initialize_wandb_logger(args, name_suffix=f'_block_{i}')
            wandb_logger.watch(imitator, log='all')
            wandb_logger.experiment.summary['replaced_block_error'] = replaced_block_error
            trainer = initialize_trainer(args.env, args.opt, wandb_logger, callbacks)
            trainer.fit(imitator, datamodule=datamodule)
            unwatch_model(imitator)


if __name__ == '__main__':
    main()
