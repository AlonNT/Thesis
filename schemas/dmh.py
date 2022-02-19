from typing import List, Union, Literal

from pydantic import root_validator
from pydantic.types import PositiveInt

from schemas.architecture import ArchitectureArgs
from schemas.data import DataArgs
from schemas.environment import EnvironmentArgs
from schemas.optimization import OptimizationArgs
from schemas.utils import ImmutableArgs, MyBaseModel, NonNegativeFloat


class DMHArgs(ImmutableArgs):

    #: Whether to imitate network performance using knn-estimator.
    imitate_with_knn: bool = False

    # #: Whether to imitate network performance using knn-estimator.  TODO consider working on this mode as well.
    # imitate_with_locally_linear_model: bool = False

    #: Whether to enable estimating intrinsic-dimension.
    estimate_intrinsic_dimension: bool = False

    #: Whether to estimate the intrinsic-dimension on the patches or on the images.
    estimate_dim_on_patches: bool = True
    estimate_dim_on_images: bool = False

    linear_regions_calculator: bool = False

    #: The minimal and maximal values of k to average the intrinsic-dimension estimate and get $\\hat(m)$ (see paper)
    k1: PositiveInt = 10
    k2: PositiveInt = 20

    #: Indicator to shuffle the patches/images before calculating the intrinsic-dimension.
    shuffle_before_estimate: bool = False

    #: Number of patches to uniformly sample (which might get clustered later).
    n_patches: Union[PositiveInt, List[PositiveInt]] = 262144

    #: How many clusters to have in the final patches' dictionary.
    n_clusters: Union[PositiveInt, List[PositiveInt]] = 1024

    #: If it's true, the patches will NOT be taken from the dataset,
    #: they will be uniformly sampled from [-1,+1]
    random_uniform_patches: Union[bool, List[bool]] = False

    #: If it's true, the patches will NOT be taken from the dataset,
    #: they will be uniformly sampled from [-1,+1]
    random_gaussian_patches: Union[bool, List[bool]] = False

    #: The k-th nearest-neighbor will be used for the k-NN imitator.
    k: Union[PositiveInt, List[PositiveInt]] = 1

    #: The regularization factor (a.k.a. lambda) of the whitening matrix.
    use_whitening: Union[bool, List[bool]] = False

    #: If it's false, use PCA-whitening. Otherwise, use ZCA whitening (which is a rotation of the PCA-whitening).
    zca_whitening: Union[bool, List[bool]] = False

    #: If it's true, calculates whitening from the sampled patches, and not from all patches in the dataset.
    calc_whitening_from_sampled_patches: Union[bool, List[bool]] = False

    #: The regularization factor (a.k.a. lambda) of the whitening matrix.
    whitening_regularization_factor: Union[NonNegativeFloat, List[NonNegativeFloat]] = 0.001

    #: Whether to normalize the patches to unit vectors (divide by its l2 norm).
    #: This cause the metric between patches to be minimal-angle instead of euclidean-distance.
    normalize_patches_to_unit_vectors: Union[bool, List[bool]] = False

    dataset_type_for_patches_dictionary: Literal['aug', 'no_aug', 'clean'] = 'aug'

    #: Whether to use faiss in calculating nearest-neighbors. If it's false native PyTorch will be used.
    use_faiss: bool = False

    #: Whether to use angles as the metric
    #: (done by normalizing the vectors to be unit-vectors before calculating euclidean distance).
    use_angles: bool = False

    #: 'none' means simply mean will be used to reduce the k's vectors to a single vector, not a linear function.
    #: 'partial' means a linear function (from k values to 1) will be used instead of mean over the k vectors.
    #: 'full' means a linear function (from k*C values to C) will be used instead of mean over the k vectors.
    use_linear_function: Literal['none', 'full', 'partial'] = 'none'

    @root_validator
    def validate_estimate_dim_on_images_or_patches(cls, values):
        assert values['estimate_dim_on_patches'] != values['estimate_dim_on_images'], "Exactly one should be given."
        return values

    @root_validator
    def validate_k1_and_k2(cls, values):
        assert values['k1'] < values['k2']  # <= values['n_patches']
        return values

    # @root_validator
    # def validate_mutually_exclusive(cls, values):
    #     assert sum(int(values[k]) for k in ['imitate_with_knn',
    #                                         'imitate_with_locally_linear_model']) <= 1, "Must be mutually exclusive"
    #     return values


class Args(MyBaseModel):
    opt = OptimizationArgs()
    arch = ArchitectureArgs()
    env = EnvironmentArgs()
    data = DataArgs()
    dmh = DMHArgs()
