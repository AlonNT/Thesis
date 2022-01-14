from typing import Optional
from pydantic import validator, PositiveInt
from schemas.utils import ImmutableArgs


class DataArgs(ImmutableArgs):

    dataset_name: str = 'CIFAR10'
    n_channels: Optional[PositiveInt] = None
    spatial_size: Optional[PositiveInt] = None

    #: Whether to use random cropping which is padding of 4 followed by random crop.
    random_crop: bool = True

    #: Whether to use random horizontal flipping (with probability 0.5).
    random_horizontal_flip: bool = True

    #: Whether to normalize of the values to a unit gaussian.
    normalization_to_unit_gaussian: bool = False

    #: Whether to normalize of the values to the interval [-1,+1].
    normalization_to_plus_minus_one: bool = True

    #: Indicator to shuffle the input images pixels before feeding to the neural-network.
    shuffle_images: bool = False

    #: Indicator to keep the RGB triplet intact when shuffling the image:
    #: sample a permutation from (32*32)! and not from (3*32*32)!
    keep_rgb_triplets_intact: bool = False

    @validator('n_channels', always=True, pre=True)
    def calculate_n_channels(cls, v, values):
        n_channels = 3 if (values['dataset_name'] == 'CIFAR10') else 1
        assert (v is None) or (v == n_channels), \
            f"n_channels is {v}, it should be None or {n_channels} for the dataset {values['dataset_name']}"
        return n_channels

    @validator('spatial_size', always=True, pre=True)
    def calculate_spatial_size(cls, v, values):
        spatial_size = 32 if (values['dataset_name'] == 'CIFAR10') else 28
        assert (v is None) or (v == spatial_size), \
            f"spatial_size is {v}, it should be None or {spatial_size} for the dataset {values['dataset_name']}"
        return spatial_size
