from typing import Literal, Optional
from pydantic import validator
from pydantic.types import PositiveInt

from schemas.utils import ImmutableArgs, NonNegativeInt, NonOneFraction
from vgg import configs


DEFAULT_PRETRAINED_PATHS = {model_name: f'alonnt/thesis/{model_name}:best' for model_name in configs.keys()}


class ArchitectureArgs(ImmutableArgs):

    #: The model name for the network architecture.
    model_name: str = 'VGG11c'

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
