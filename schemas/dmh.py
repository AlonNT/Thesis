from pydantic import root_validator
from pydantic.types import PositiveInt

from schemas.architecture import ArchitectureArgs
from schemas.data import DataArgs
from schemas.environment import EnvironmentArgs
from schemas.optimization import OptimizationArgs
from schemas.utils import ImmutableArgs, MyBaseModel


class IntDimEstArgs(ImmutableArgs):

    #: Whether to enable estimating intrinsic-dimension.
    estimate_intrinsic_dimension: bool = False

    #: Number of patches/images to sample uniformly at random to estimate the intrinsic dimension.
    n: PositiveInt = 8192

    #: Whether to estimate the intrinsic-dimension on the patches or on the images.
    estimate_dim_on_patches: bool = True
    estimate_dim_on_images: bool = False

    #: The minimal and maximal values of k to average the intrinsic-dimension estimate and get $\\hat(m)$ (see paper)
    k1: PositiveInt = 10
    k2: PositiveInt = 20

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

    #: Whether to imitate network performance using knn-estimator.
    imitate_with_knn: bool = False

    #: Whether to imitate network performance using knn-estimator.
    imitate_with_locally_linear_model: bool = False

    #: Number of patches to uniformly sample to perform clustering.
    n_patches: PositiveInt = 65536

    #: How many clusters to have in the final patches dictionary.
    n_clusters: PositiveInt = 1024

    #: If it's true, the patches will NOT be taken from the dataset,
    #: they will be uniformly sampled from [-1,+1]
    random_patches: bool = False

    #: The k-th nearest-neighbor will be used for the imitation using locally linear model.
    k: PositiveInt = 1

    @root_validator
    def validate_mutually_exclusive(cls, values):
        assert not (values['imitate_with_knn'] and values['imitate_with_locally_linear_model']), \
            "Must be mutually exclusive"
        return values


class Args(MyBaseModel):
    opt = OptimizationArgs()
    arch = ArchitectureArgs()
    env = EnvironmentArgs()
    data = DataArgs()
    int_dim_est = IntDimEstArgs()
    imitation = ImitationArgs()

    @root_validator
    def validate_imitate_on_pretrained_only(cls, values):
        if values['imitation'].imitate_with_locally_linear_model or values['imitation'].imitate_with_knn:
            values['arch'].use_pretrained = True
        return values
