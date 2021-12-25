from typing import List

from pydantic.types import PositiveInt

from schemas.environment import EnvironmentArgs
from schemas.data import DataArgs
from schemas.utils import MyBaseModel


class Args(MyBaseModel):
    env = EnvironmentArgs()
    data = DataArgs()

    extrinsic_dimension: List[PositiveInt] = [3, 100, 1000]
    gaussian_dimension: List[PositiveInt] = [2, 10, 20]
    noise_std: List[float] = [0, 0.01, 0.1, 1]
    n_points: PositiveInt = 10000
    start_k: PositiveInt = 5
    k1: PositiveInt = 10
    k2: PositiveInt = 20
