from typing import Literal, Optional
from pydantic import validator, PositiveInt
from schemas.utils import ImmutableArgs


class DataArgs(ImmutableArgs):

    dataset_name: Literal['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'ImageNet'] = 'CIFAR10'
    n_channels: Optional[PositiveInt] = None
    spatial_size: Optional[PositiveInt] = None
    n_classes: Optional[PositiveInt] = None

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
    keep_rgb_triplets_intact: bool = True

    #: If it's true, a fixed permutation will be used every time and not a fresh new sampled permutation.
    fixed_permutation: bool = True

    #: The firectory containing the data (for small dataset, i.e. not ImageNet, 
    #: the data will be downloaded to this directory if it's not already there).
    data_dir: str = "./data"

    @validator('n_channels', always=True, pre=True)
    def calculate_n_channels(cls, v, values):
        dataset_name = values['dataset_name']
        if dataset_name in ['CIFAR10', 'CIFAR100', 'ImageNet']:
            n_channels = 3
        elif dataset_name in ['MNIST', 'FashionMNIST']:
            n_channels = 1
        else:
            raise ValueError(f'{dataset_name=} is not supported.')

        assert (v is None) or (v == n_channels), \
            f"n_channels is {v}, it should be None or {n_channels} for the dataset {values['dataset_name']}"
            
        return n_channels

    @validator('spatial_size', always=True, pre=True)
    def calculate_spatial_size(cls, v, values):
        dataset_name = values['dataset_name']
        if dataset_name in ['CIFAR10', 'CIFAR100']:
            spatial_size = 32
        elif dataset_name in ['MNIST', 'FashionMNIST']:
            spatial_size = 28
        elif dataset_name in ['ImageNet']:
            spatial_size = 224
        else:
            raise ValueError(f'{dataset_name=} is not supported.')

        assert (v is None) or (v == spatial_size), \
            f"spatial_size is {v}, it should be None or {spatial_size} for the dataset {values['dataset_name']}"

        return spatial_size

    @validator('n_classes', always=True, pre=True)
    def calculate_n_classes(cls, v, values):
        dataset_name = values['dataset_name']
        if dataset_name in ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST']:
            n_classes = 10
        elif dataset_name in ['ImageNet']:
            n_classes = 1000
        else:
            raise ValueError(f'{dataset_name=} is not supported.')

        assert (v is None) or (v == n_classes), \
            f"n_classes is {v}, it should be None or {n_classes} for the dataset {values['dataset_name']}"

        return n_classes
