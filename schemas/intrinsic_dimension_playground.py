from typing import List, Union

from pydantic.types import PositiveInt

from schemas.environment import EnvironmentArgs
from schemas.data import DataArgs
from schemas.utils import ImmutableArgs, MyBaseModel


class IntDimArgs(ImmutableArgs):
    k1: PositiveInt = 10
    k2: PositiveInt = 20
    n_points: Union[PositiveInt, List[PositiveInt]] = 1048576
    whitening_regularization_factor: float = 0.001
    zca_whitening: bool = False

    shuffle_before_estimate: bool = False

    gaussian_playground: bool = False
    extrinsic_dimension: List[PositiveInt] = [3, 100, 1000]
    gaussian_dimension: List[PositiveInt] = [2, 10, 20]
    noise_std: List[float] = [0, 0.01, 0.1, 1]
    start_k: PositiveInt = 10

    patch_size: Union[PositiveInt, List[PositiveInt]] = 5
    normalize_patches_to_unit_vectors: bool = False
    
    cifar_mle: bool = False
    
    cifar_elbow: bool = False
    min_n_centroids: PositiveInt = 2
    max_n_centroids: PositiveInt = 100

    linear_regions: bool = False
    model_path: str = 'alonnt/thesis/model-13dgm3zv:v0'  # 1024 channels


class Args(MyBaseModel):
    env = EnvironmentArgs()
    data = DataArgs()
    int_dim = IntDimArgs()
