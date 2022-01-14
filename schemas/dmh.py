from math import ceil
from typing import Optional

from pydantic import root_validator
from pydantic.types import PositiveInt

from schemas.architecture import ArchitectureArgs
from schemas.data import DataArgs
from schemas.environment import EnvironmentArgs
from schemas.optimization import OptimizationArgs
from schemas.utils import ImmutableArgs, MyBaseModel, ProperFraction, NonNegativeFloat, NonNegativeInt


class DMHArgs(ImmutableArgs):

    #: Whether to train a (shallow) locally linear network.
    train_locally_linear_network: bool = False

    #: Whether to imitate network performance using knn-estimator.
    imitate_with_knn: bool = False

    #: Whether to imitate network performance using knn-estimator.
    imitate_with_locally_linear_model: bool = False

    #: Whether to enable estimating intrinsic-dimension.
    estimate_intrinsic_dimension: bool = False

    #: Whether to estimate the intrinsic-dimension on the patches or on the images.
    estimate_dim_on_patches: bool = True
    estimate_dim_on_images: bool = False

    #: The minimal and maximal values of k to average the intrinsic-dimension estimate and get $\\hat(m)$ (see paper)
    k1: PositiveInt = 10
    k2: PositiveInt = 20

    #: Indicator to shuffle the patches/images before calculating the intrinsic-dimension.
    shuffle_before_estimate: bool = False

    #: Number of patches to uniformly sample (which might get clustered later).
    n_patches: PositiveInt = 262144

    #: How many clusters to have in the final patches' dictionary.
    n_clusters: PositiveInt = 1024

    #: If it's true, the patches will NOT be taken from the dataset,
    #: they will be uniformly sampled from [-1,+1]
    random_uniform_patches: bool = False

    #: If it's true, the patches will NOT be taken from the dataset,
    #: they will be uniformly sampled from [-1,+1]
    random_gaussian_patches: bool = False

    #: The k-th nearest-neighbor will be used for the k-NN imitator, or in the locally linear model.
    k: PositiveInt = 64

    #: The k-th nearest-neighbor as a fraction of the total amount of clusters
    k_fraction: Optional[ProperFraction] = None

    #: If it's true, when calculating k-nearest-neighbors there will be ones in the indices of the neighbors
    #: 1, 2, ..., k. If it's false, there will be a single one in the index of the k-th nearest-neighbor.
    up_to_k: bool = True

    #: Patch size.
    patch_size: PositiveInt = 5

    #: Amount of padding to use in the k-nearest-neighbors embedding.
    padding: NonNegativeInt = 0

    #: Stride to use in the k-nearest-neighbors embedding.
    stride: PositiveInt = 1

    #: Use convolution layer multiplied by the patch-based embedding.
    #: If it's false the resulting model is the same as Thiry et al.
    use_conv: bool = True

    #: Initialize the patches dictionary randomly from the same random distribution as PyTorch default for Conv2D.
    random_embedding: bool = False

    #: Taken from page 4 from Coates et al (2011)
    #: An Analysis of Single-Layer Networks in Unsupervised Feature Learning
    #: https://cs.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
    #: This activation function outputs 0 for any patch where the distance to the centroid  is “above average”.
    #: In practice, this means that roughly half of the features will be set to 0.
    kmeans_triangle: bool = False

    #: If it's true, the embedding with be replaced with a Conv->ReLU
    replace_embedding_with_regular_conv_relu: bool = False

    #: If it's true, the embedding will have gradients and will change during training.
    learnable_embedding: bool = False

    #: The regularization factor (a.k.a. lambda) of the whitening matrix.
    use_whitening: bool = True

    #: If it's false, use PCA-whitening. Otherwise, use ZCA whitening (which is a rotation of the PCA-whitening).
    zca_whitening: bool = False

    #: If it's true, calculates whitening from the sampled patches, and not from all patches in the dataset.
    calc_whitening_from_sampled_patches: bool = False

    #: The regularization factor (a.k.a. lambda) of the whitening matrix.
    whitening_regularization_factor: NonNegativeFloat = 0.001

    #: The regularization factor (a.k.a. lambda) of the whitening matrix.
    normalize_patches_to_unit_vectors: bool = False

    #: Whether to use avg-pool after the embedding.
    use_avg_pool: bool = True
    pool_size: PositiveInt = 4
    pool_stride: PositiveInt = 4

    #: Whether to use adaptive-avg-pool after the avg-pool.
    use_adaptive_avg_pool: bool = False
    adaptive_pool_output_size: PositiveInt = 6

    #: Whether to use batch-norm after the embedding or not.
    use_batch_norm: bool = True

    #: Whether to decompose the final linear layer into 1x1 convolution followed by an actual linear layer.
    use_bottle_neck: bool = True
    bottle_neck_dimension: PositiveInt = 128
    bottle_neck_kernel_size: PositiveInt = 1
    use_relu_after_bottleneck: bool = False

    #: If depth = 2 it means that we build one model on top of the other.
    depth: int = 1

    #: Whether the input embedding to the deep model should have a residual connection (addition / concatenation).
    residual_add: bool = False
    residual_cat: bool = False

    #: Number of values to output for each patch's linear-classifier.
    c: PositiveInt = 1

    @root_validator
    def set_k_from_k_fraction(cls, values):
        if values['k_fraction'] is not None:
            assert values['k'] == DMHArgs.__fields__['k'].default, "If you give k_fraction, don't give k."
            values['k'] = ceil(values['k_fraction'] * values['n_clusters'])
        return values

    @root_validator
    def validate_estimate_dim_on_images_or_patches(cls, values):
        assert values['estimate_dim_on_patches'] != values['estimate_dim_on_images'], "Exactly one should be given."
        return values

    @root_validator
    def validate_k1_and_k2(cls, values):
        assert values['k1'] < values['k2'] <= values['n_patches']
        return values

    @root_validator
    def validate_mutually_exclusive(cls, values):
        assert sum(int(values[k]) for k in ['imitate_with_knn',
                                            'imitate_with_locally_linear_model',
                                            'train_locally_linear_network']) <= 1, "Must be mutually exclusive"
        return values


class Args(MyBaseModel):
    opt = OptimizationArgs()
    arch = ArchitectureArgs()
    env = EnvironmentArgs()
    data = DataArgs()
    dmh = DMHArgs()
