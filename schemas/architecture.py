from typing import Literal, Optional, List, Union
from pydantic import validator
from pydantic.types import PositiveInt

from schemas.utils import ImmutableArgs, NonNegativeInt, NonOneFraction, NonNegativeFloat
from vgg import configs


DEFAULT_PRETRAINED_PATHS = {model_name: f'alonnt/thesis/{model_name}:best' for model_name in configs.keys()}


class ArchitectureArgs(ImmutableArgs):

    #: The model name for the network architecture.
    model_name: str = 'VGG11c'

    #: Whether to put the (avg/max) pool layers as separate blocks, or in the end of the previous conv block.
    pool_as_separate_blocks: bool = True

    #: If it's true - shuffle the output of each block in the network.
    #: The output will be shuffled spatially only, meaning that the channels dimension will stay intact.
    #: For example, if the output is of shape 64x28x28 a random permutation from all (28*28)! possibilities
    #: is sampled and applied to the input tensor.
    shuffle_blocks_output: Union[bool, List[bool]] = False

    #: If it's true, shuffle the spatial locations only and the channels dimension will stay intact.
    spatial_shuffle_only: Union[bool, List[bool]] = True

    #: If it's true - use a fixed permutation per block in the network and not sample a new one each time.
    fixed_permutation_per_block: Union[bool, List[bool]] = True

    #: Use pretrained model, or train from scratch.
    use_pretrained: bool = False

    #: Pretrained model path (in wandb).
    pretrained_path: Optional[str] = None

    #: How many hidden layers the final MLP at the end of the convolution blocks.
    mlp_n_hidden_layers: NonNegativeInt = 1

    #: Dimension of each hidden layer the final MLP at the end of the convolution blocks.
    mlp_hidden_dim: Union[PositiveInt, List[PositiveInt]] = 128

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
    use_batch_norm: Union[bool, List[bool]] = True

    #: If it's greater than zero, adds a 1x1 convolution layer ("bottleneck") in the end of the block.
    bottle_neck_dimension: Union[NonNegativeInt, List[NonNegativeInt]] = 0

    #: The size for the bottleneck layer(s).
    bottle_neck_kernel_size: Union[PositiveInt, List[PositiveInt]] = 1

    #: Padding mode for the convolution layers.
    padding_mode: Literal['zeros', 'circular'] = 'zeros'

    #: Coefficients for lasso regularizers (where zero means no regularization is applied).
    #: There could be potentially different values for different layers in the model.
    lasso_regularizer_coefficient: Union[NonNegativeFloat, List[NonNegativeFloat]] = 0

    #: The argument $\beta$ in "beta-lasso" algorithm from "Towards Learning Convolutions from Scratch".
    #: There could be potentially different values for different layers in the model.
    beta_lasso_coefficient: Union[NonNegativeFloat, List[NonNegativeFloat]] = 0

    #: The scaling factor for the basic CNN (following $\alpha$ in "Towards Learning Convolutions from Scratch").
    alpha: PositiveInt = 32

    @validator('model_name', always=True)
    def validate_model_name(cls, v):
        allowed_values = list(configs.keys()) + ['mlp'] + [f'{a}-{b}' for a in ['S', 'D'] for b in ['CONV', 'FC']]
        assert v in allowed_values, f"model_name {v} is not supported, should be one of {allowed_values}"
        return v

    # TODO Commented-out, because for some reason `model_name` is not in `values` (e.g. KeyError: 'model_name').
    #      Handle this when we'll want to use pretrained models.
    # @validator('pretrained_path', always=True)
    # def set_pretrained_path(cls, v, values):
    #     if v is None:
    #         v = DEFAULT_PRETRAINED_PATHS.get(values['model_name'])
    #     return v
