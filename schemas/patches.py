import datetime
import torch
import flatten_dict

from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel, Extra, validator
from pydantic.types import PositiveInt, ConstrainedFloat, DirectoryPath

from consts import DATETIME_STRING_FORMAT


class NonNegativeFloat(ConstrainedFloat):
    ge = 0


class ProperFraction(ConstrainedFloat):
    gt = 0
    lt = 1


class ImmutableArgs(BaseModel):
    class Config:
        allow_mutation = True
        extra = Extra.forbid


class OptimizationArgs(ImmutableArgs):

    #: Number of epochs to train.
    epochs: PositiveInt = 80

    #: Mini batch size to use in each training-step.
    batch_size: PositiveInt = 128

    #: Momentum to use in SGD optimizer.
    momentum: ProperFraction = 0.9

    #: Amount of weight decay (regularization).
    weight_decay: NonNegativeFloat = 0

    # The initial learning-rate which might later be decayed.
    learning_rate: ProperFraction = 0.003

    #: Decay the learning-rate at these steps by a factor of `learning_rate_decay_gamma`.
    learning_rate_decay_steps: List[PositiveInt] = [50, 75]

    #: The factor gamma to multiply the learning-rate at the decay steps.
    learning_rate_decay_gamma: ProperFraction = 0.1

    @validator('learning_rate_decay_steps', each_item=True)
    def validate_learning_rate_decay_steps_below_epochs(cls, v, values):
        assert v < values['epochs'], "Each decay step must be lower than the total number of epoch."
        return v

    @validator('learning_rate_decay_steps')
    def validate_learning_rate_decay_steps_are_ascending(cls, v):
        assert all(v[i] <= v[i+1] for i in range(len(v)-1)), "Decay steps should be ascending."
        return v


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

    @validator('k_neighbors', always=True, pre=True)
    def calculate_k_neighbors(cls, v, values):
        assert v is None, 'The argument k_neighbors should not be given, ' \
                          'it will be calculated automatically from k_neighbors_fraction and n_patches.'
        return int(values['k_neighbors_fraction'] * float(values['n_patches']))


class EnvironmentArgs(ImmutableArgs):

    #: On which device to train.
    device: Literal['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7'] = 'cpu'

    #: Output path for the experiment - a sub-directory named with the data and time will be created within.
    path: DirectoryPath = './experiments'

    #: How many iterations between each training log.
    log_interval: PositiveInt = 100

    @validator('path', always=True)
    def create_out_dir(cls, v: Path):
        datetime_string = datetime.datetime.now().strftime(DATETIME_STRING_FORMAT)
        out_dir = v / datetime_string
        out_dir.mkdir(exist_ok=True)  # exist_ok because this validator is being called multiple times (I think)
        return out_dir

    @validator('device')
    def validate_device_exists(cls, v):
        if v.startswith('cuda:'):
            assert torch.cuda.is_available(), f"CUDA is not available, so can't use device {v}"
            assert int(v[-1]) < torch.cuda.device_count(), f"GPU index {v[-1]} is higher than the number of GPUS."
        return v


class DataArgs(ImmutableArgs):
    #: Whether to normalize of the values to a unit gaussian.
    normalization_to_unit_gaussian: bool = True

    #: Whether to use random cropping which is padding of 4 followed by random crop.
    random_crop: bool = True

    #: Whether to use random horizontal flipping (with probability 0.5).
    random_horizontal_flip: bool = True

    #: Whether to normalize of the values to the interval [-1,+1].
    normalization_to_plus_minus_one: bool = False


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
