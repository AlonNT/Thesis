from pydantic import root_validator
from pydantic.types import PositiveInt

from schemas.architecture import ArchitectureArgs
from schemas.data import DataArgs
from schemas.environment import EnvironmentArgs
from schemas.optimization import OptimizationArgs
from schemas.utils import ImmutableArgs, MyBaseModel


class IntDimEstArgs(ImmutableArgs):
    estimate_intrinsic_dimension: bool = True

    #: Number of patches/images to sample uniformly at random to estimate the intrinsic dimension.
    n: PositiveInt = 8192

    #: Whether to estimate the intrinsic-dimension on the patches or on the images.
    estimate_dim_on_patches: bool = True
    estimate_dim_on_images: bool = False

    #: The minimal and maximal values of k to average the intrinsic-dimension estimate and get $\\hat(m)$ (see paper)
    k1: PositiveInt = 10
    k2: PositiveInt = 20

    #: Indicator to plot graphs of the k-th intrinsic-dimension estimate for different k's.
    log_graphs: bool = True

    #: Indicator to shuffle the patches/images before calculating the intrinsic-dimension.
    shuffle_before_estimate: bool = False

    @root_validator
    def validate_estimate_dim_on_images_or_patches(cls, values):
        assert values['estimate_dim_on_patches'] != values['estimate_dim_on_images'], "Exactly one should be given."
        return values

    @root_validator
    def validate_k1_and_k2(cls, values):
        assert values['k1'] < values['k2'] <= values['n']
        return values


class ImitationArgs(ImmutableArgs):
    imitate_with_knn: bool = False
    n_clusters: PositiveInt = 1024


class Args(MyBaseModel):
    opt = OptimizationArgs()
    arch = ArchitectureArgs()
    env = EnvironmentArgs()
    data = DataArgs()
    int_dim_est = IntDimEstArgs()
    imitation = ImitationArgs()
