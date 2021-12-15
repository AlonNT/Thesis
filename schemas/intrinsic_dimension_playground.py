from typing import List

from flatten_dict import flatten_dict
from pydantic import BaseModel
from pydantic.types import PositiveInt

from schemas.environment import EnvironmentArgs
from schemas.data import DataArgs


class Args(BaseModel):
    env = EnvironmentArgs()
    data = DataArgs()

    gaussian_playground: bool = True
    extrinsic_dimension: List[PositiveInt] = [3, 100, 1000]
    gaussian_dimension: List[PositiveInt] = [2, 10, 20]
    noise_std: List[float] = [0, 0.01, 0.1, 1]
    n_points: PositiveInt = 10000
    start_k: PositiveInt = 5
    k1: PositiveInt = 10
    k2: PositiveInt = 20

    def flattened_dict(self):
        """
        Returns the arguments as a flattened dictionary, without the category name (i.e. opt, arch, env, data).
        It's assumed that there is no field with the same name among different categories.
        """
        return {k[-1]: v for k, v in flatten_dict.flatten(self.dict()).items()}
