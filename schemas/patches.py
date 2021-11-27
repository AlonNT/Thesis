from typing import Optional

import flatten_dict
from pydantic import BaseModel, validator
from pydantic.types import PositiveInt

from schemas.data import DataArgs
from schemas.environment import EnvironmentArgs
from schemas.optimization import OptimizationArgs
from schemas.utils import ImmutableArgs, ProperFraction, NonNegativeFloat


class ArchitectureArgs(ImmutableArgs):

    #: Number of channels in the convolution layer which comes after the embedding.
    n_channels: PositiveInt = 128

    #: The number of patches.
    n_patches: PositiveInt = 2048

    #: The size of the patches.
    patch_size: PositiveInt = 6

    #: Whether to use average-pooling on the patch-based-embedding.
    use_avg_pool: bool = True

    #: The size of the kernel in the convolution layer after the patch-based-embedding (a.k.a. 'bottle-neck' layer).
    conv_kernel_size: PositiveInt = 1

    #: The size of the average-pooling layer after the patch-based-embedding.
    pool_size: PositiveInt = 5

    #: The stride of the average-pooling layer after the patch-based-embedding.
    pool_stride: PositiveInt = 3

    #: which k to use in the k-nearest-neighbors, as a fraction of the total number of patches.
    k_neighbors_fraction: ProperFraction = 0.4

    #: Which k to use in the k-nearest-neighbors.
    #: Will be calculated automatically from k_neighbors_fraction and n_patches.
    k_neighbors: Optional[PositiveInt] = None

    #: If true, use whitening on the patches.
    use_whitening: bool = True

    #: The regularization factor (a.k.a. lambda) of the whitening matrix.
    whitening_regularization_factor: NonNegativeFloat = 0.001

    #: Whether to use batch normalization after the patch-based-embedding.
    use_batch_norm: bool = True

    #: Whether to use ReLU after the bottleneck layer.
    use_relu_after_bottleneck: bool = False

    #: Whether to use the negative patches as well (i.e. original patches multiplied by -1).
    #: These patches are being used as a separate network branch, as the original paper.
    add_negative_patches_as_network_branch: bool = True

    #: Whether to use the negative patches as well (i.e. original patches multiplied by -1).
    #: These patches are being concatenated to the original patches so the dictionary size is multiplied by two.
    add_negative_patches_as_more_patches: bool = False

    #: Whether to use the negative patches as well (i.e. original patches multiplied by -1).
    add_flipped_patches: bool = False

    #: Whether to use the adaptive avg-pooling on the embedding output to get spatial size 6.
    use_adaptive_avg_pool: bool = True

    #: The depth of the neural-network, if greater than 1 then multiple base networks will be stacked together.
    depth: PositiveInt = 1

    @validator('k_neighbors', always=True, pre=True)
    def calculate_k_neighbors(cls, v, values):
        assert v is None, 'The argument k_neighbors should not be given, ' \
                          'it will be calculated automatically from k_neighbors_fraction and n_patches.'
        return int(values['k_neighbors_fraction'] * float(values['n_patches']))
    
    @validator('depth', always=True)
    def validate_depth(cls, v):
        assert v in {1, 2}, "Currently only depth 1 or 2 is supported."
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
