from typing import Literal
from pydantic import root_validator
from pydantic.types import PositiveInt

from schemas.architecture import ArchitectureArgs
from schemas.data import DataArgs
from schemas.environment import EnvironmentArgs
from schemas.optimization import OptimizationArgs
from schemas.utils import ImmutableArgs, MyBaseModel, NonNegativeInt, Fraction


AUX_NET_TYPE = Literal['cnn', 'mlp']


class LayerwiseArgs(ImmutableArgs):

    # TODO: do we want this functionality?
    # #: The first trainable block index. Positive values can be used to fix first few blocks in their initial weights.
    # first_trainable_block: NonNegativeInt = False

    # ################################ Decoupled-Greedy-Learning ################################

    #: Use decoupled greedy learning to train the network.
    dgl: bool = False

    #: Type of the auxiliary networks predicting the classes scores.
    pred_aux_type: AUX_NET_TYPE = 'cnn'

    #: How many hidden layers in each auxiliary network (which is an MLP).
    aux_mlp_n_hidden_layers: NonNegativeInt = 1

    #: Dimension of each hidden layer in each auxiliary network (which is an MLP).
    aux_mlp_hidden_dim: PositiveInt = 128

    # ################################ Self-Supervised Local Loss ################################

    #: Use self-supervised local loss (predict the image, or the shifted image).
    ssl: bool = False

    #: Shift the target images to be produced by the SSL auxiliary networks.
    shift_ssl_labels: bool = False

    #: Whether to up-sample SSL predictions or leave them down-sampled.
    upsample: bool = False

    #: Weight of the prediction loss when using both prediction-loss and SSL-loss.
    pred_loss_weight: Fraction = 0.9

    #: Weight of the SSL loss when using both prediction-loss and SSL-loss.
    ssl_loss_weight: Fraction = 0.1

    # ################################ Arguments for optimizing with last gradient ################################

    #: The last trainable block index. Positive values can be used to fix last few blocks in their initial weights.
    #: Whether to use direct global gradient.
    is_direct_global: bool = False

    #: Whether to use last gradient in each intermediate module.
    #: Similar theoretically to `is_direct_global` but implemented quite differently.
    use_last_gradient: bool = False

    #: Weight of the last gradient to be used in each intermediate gradient calculator.
    #: The intermediate gradient will be
    #: (1-last_gradient_weight)*original_gradient + last_gradient_weight*last_layer_gradient.
    last_gradient_weight: Fraction = 0.5

    @root_validator
    def validate_ssl_or_lg_comes_with_dgl(cls, values):
        if (values['ssl'] or values['is_direct_global']) and (not values['dgl']):
            assert False, "When training with local self-supervised loss or direct global loss, " \
                          "one should also give --dgl (because it's decoupled modules as well). " \
                          "You can give zero loss weight for the score's loss to only use ssl."
        return values

    @root_validator
    def validate_ssl_and_lg_can_not_come_together(cls, values):
        assert not (values['is_direct_global'] and values['ssl']), \
            "Can not use both direct global loss and ssl at the moment."
        return values

    @root_validator
    def validate_is_direct_global_and_last_gradient_can_not_come_together(cls, values):
        assert not (values['is_direct_global'] and values['use_last_gradient']), \
            "Can not use both is_direct_global and use_last_gradient at the moment."
        return values

    @root_validator
    def validate_ssl_loss_weight_and_pred_loss_weight_should_sum_to_one(cls, values):
        assert values['pred_loss_weight'] + values['ssl_loss_weight'] == 1, \
            "pred_loss_weight and ssl_loss_weight should sum to 1."
        return values


class Args(MyBaseModel):
    opt = OptimizationArgs()
    arch = ArchitectureArgs()
    env = EnvironmentArgs()
    data = DataArgs()
    layerwise = LayerwiseArgs()

    @root_validator
    def validate_circular_padding_in_shifted_ssl_training(cls, values):
        assert not (values['layerwise'].shift_ssl_labels and values['arch'].padding_mode), \
            "When using shifted images predictions, it's better to use circular-padding and not zero-padding."
        return values
