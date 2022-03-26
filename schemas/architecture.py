from typing import Literal, Optional, List, Union
from pydantic import validator
from pydantic.types import PositiveInt

from schemas.utils import ImmutableArgs, NonNegativeInt, NonOneFraction
from vgg import configs


DEFAULT_PRETRAINED_PATHS = {model_name: f'alonnt/thesis/{model_name}:best' for model_name in configs.keys()}


class ArchitectureArgs(ImmutableArgs):

    #: The model name for the network architecture.
    model_name: str = 'VGG11c'

    #: Whether to put the (avg/max) pool layers as separate blocks, or in the end of the previous conv block.
    pool_as_separate_blocks: bool = True

    #: Use pretrained model, or train from scratch.
    use_pretrained: bool = False

    #: Pretrained model path (in wandb).
    pretrained_path: Optional[str] = None

    #: How many hidden layers the final MLP at the end of the convolution blocks.
    final_mlp_n_hidden_layers: NonNegativeInt = 1

    #: Dimension of each hidden layer the final MLP at the end of the convolution blocks.
    final_mlp_hidden_dim: PositiveInt = 128

    #: Dropout probability (will be added after each non-linearity).
    dropout_prob: NonOneFraction = 0

    input_channels: PositiveInt = 3
    input_spatial_size: PositiveInt = 32

    #: The kernel-size to use in each convolution-layer.
    kernel_size: Union[PositiveInt, List[PositiveInt]] = 3

    #: Stride to use in the convolution-layer.
    stride: Union[PositiveInt, List[PositiveInt]] = 1

    #: The padding amount to use in each convolution-layer.
    padding: Union[NonNegativeInt, List[NonNegativeInt]] = 1

    #: The pooling size and stride to use in the AvgPool / MaxPool layers.
    pool_size: Union[PositiveInt, List[PositiveInt]] = 4
    pool_stride: Union[PositiveInt, List[PositiveInt]] = 4

    #: Whether to use batch-normalization layer after the Conv -> ReLU (and possible pool) part in the block.
    use_batch_norm: Union[bool, List[bool]] = False

    #: If it's greater than zero, adds a 1x1 convolution layer ("bottleneck") in the end of the block.
    bottle_neck_dimension: Union[NonNegativeInt, List[NonNegativeInt]] = 0

    #: The size for the bottleneck layer(s).
    bottle_neck_kernel_size: Union[PositiveInt, List[PositiveInt]] = 1

    #: Padding mode for the convolution layers.
    padding_mode: Literal['zeros', 'circular'] = 'zeros'

    @validator('model_name', always=True)
    def validate_model_name(cls, v):
        assert v == 'mlp' or v in configs.keys(), f"model_name {v} is not supported, " \
                                                  f"should be 'mlp' or one of {list(configs.keys())}"
        return v

    @validator('pretrained_path', always=True)
    def set_pretrained_path(cls, v, values):
        if v is None:
            v = DEFAULT_PRETRAINED_PATHS.get(values['model_name'])
        return v
