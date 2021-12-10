from typing import Literal

import flatten_dict
from pydantic import BaseModel, validator, root_validator
from pydantic.types import PositiveInt

from schemas.data import DataArgs
from schemas.environment import EnvironmentArgs
from schemas.optimization import OptimizationArgs
from schemas.utils import ImmutableArgs, NonNegativeInt, NonOneFraction
from vgg import configs


class ArchitectureArgs(ImmutableArgs):

    #: The model name for the network architecture.
    model_name: str = 'VGG11c'

    #: How many hidden layers the final MLP at the end of the convolution blocks.
    final_mlp_n_hidden_layers: NonNegativeInt = 1

    #: Dimension of each hidden layer the final MLP at the end of the convolution blocks.
    final_mlp_hidden_dim: PositiveInt = 128

    #: Dropout probability (will be added after each non linearity).
    dropout_prob: NonOneFraction = 0

    #: Padding mode for the convolution layers.
    padding_mode: Literal['zeros', 'circular'] = 'zeros'

    #: Number of patches to sample uniformly at random to estimate the intrinsic dimension.
    n_patches: PositiveInt = 16384

    #: The minimal and maximal values of k to average the intrinsic-dimension estimate and get $\\hat(m)$ (see paper)
    k1: PositiveInt = 10
    k2: PositiveInt = 20

    #: Indicator to plot graphs of the k-th intrinsic-dimension estimate for different k's.
    plot_graphs: bool = False

    #: Indicator to shuffle the patches before calculating the intrinsic-dimension
    shuffle_patches: bool = False

    @root_validator
    def validate_k1_and_k2(cls, values):
        assert values['k1'] < values['k2'] <= values['n_patches']
        return values

    @validator('model_name', always=True)
    def validate_model_name(cls, v):
        assert v in configs.keys(), f"model_name {v} is not supported, should be one of {list(configs.keys())}"
        return v


class Args(BaseModel):
    opt = OptimizationArgs()
    arch = ArchitectureArgs()
    env = EnvironmentArgs()
    data = DataArgs()

    def flattened_dict(self):
        """
        Returns the arguments as a flattened dictionary, without the category name (i.e. opt, arch, env, data).
        It's assumed that there is no field with the same name among different categories.
        """
        return {k[1]: v for k, v in flatten_dict.flatten(self.dict()).items()}
