import torch
import torch.nn as nn

from typing import List, Optional, Union, Tuple

from consts import CLASSES
from utils import get_mlp

# Inspiration was taken from
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

# Configurations for the VGG models family.
# A number indicates number of channels in a convolution block, and M denotes a MaxPool layer.
configs = {
    # This is an architecture which reaches final dimensions of 4x4.
    'VGG6': [64, 128, 'M', 128, 256, 'M', 256, 512, 'M'],

    'VGGs': [8, 'M', 16, 'M', 32, 'M'],  # 's' for shallow.
    'VGGsw': [64, 'M', 128, 'M', 256, 'M'],  # 'w' for wide.
    'VGGsxw': [128, 'M', 256, 'M', 512, 'M'],  # 'x' for extra.

    # These are versions similar to the original VGG models, but with less down-sampling
    # (because it's CIFAR-10 32x32 and not ImageNet 224x224)
    'VGG8c': [64, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'VGG11c': [64, 128, 'M', 256, 256, 'M', 512, 512, 512, 512, 'M'],
    'VGG13c': [64, 64, 128, 128, 'M', 256, 256, 'M', 512, 512, 512, 512, 'M'],

    # https://github.com/anokland/local-loss/blob/master/train.py#L1276
    'VGG8b': [128, 256, 'M', 256, 512, 'M', 512, 'M', 512, 'M'],  #
    'VGG11b': [128, 128, 128, 256, 'M', 256, 512, 'M', 512, 512, 'M', 512, 'M'],

    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def get_ssl_aux_net(channels: int, image_size: Optional[int] = None, target_image_size: Optional[int] = None) -> \
        nn.Sequential:
    """
    Build an auxiliary network suited for self-supervised learning.
    This means a network which predicts an image (a tensor of shape H x W x 3).
    If image_size and target_image_size are given, the aux-net first upsample using a stack of ConvTranspose2d.

    :param channels: Number of channels in the input tensor to the return aux-net.
    :param image_size: If given, this is the source image size (to be upsampled).
    :param target_image_size: If given, this is the target image size (to be upsampled to).
    :return: The auxiliary network.
    """
    layers: List[nn.Module] = list()

    assert (image_size is None) == (target_image_size is None), \
        "image_size and target_image_size should be both None (in which case no up-sampling is done), " \
        "or both not None (in which case upsampling is done from image_size to target_image_size)."

    in_channels = channels
    if image_size is not None:
        while image_size != target_image_size:
            # Upsample by doubling the spatial size and halving the number of channels.
            out_channels = in_channels // 2
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            image_size *= 2
            in_channels = out_channels

    # Convolution layer with 1x1 kernel to change channels to 3.
    layers.append(nn.Conv2d(in_channels, out_channels=3, kernel_size=1))

    return nn.Sequential(*layers)


def get_blocks(config: List[Union[int, str]],
               aux_mlp_n_hidden_layers: int = 1,
               aux_mlp_hidden_dim: int = 1024,
               dropout_prob: float = 0,
               padding_mode: str = 'zeros',
               upsample: bool = False) -> Tuple[List[nn.Module], List[nn.Module], List[nn.Module], int]:
    """
    Return a list of `blocks` which constitute the whole network,
    as well as lists of the auxiliary networks (both for prediction and for SSL).
    Each block is a sequence of several layers (Conv, BatchNorm, ReLU, MaxPool2d and Dropout).

    :param config: A list of integers / 'M' describing the architecture (see examples in the top of the file).
    :param aux_mlp_n_hidden_layers: Number of hidden layers in each prediction auxiliary network.
    :param aux_mlp_hidden_dim: The dimension of each hidden layer in each prediction auxiliary network.
    :param dropout_prob: When positive, add dropout after each non-linearity.
    :param padding_mode: Should be 'zeros' or 'circular' indicating the padding mode of each Conv layer.
    :param upsample: If true, upsample each SSL auxiliary network output to 32 x 32.
    :return: The blocks, the prediction auxiliary networks, the prediction auxiliary networks,
             and the dimension of the last layer (will be useful when feeding into a liner layer later).
    """
    blocks: List[nn.Module] = list()
    auxiliary_nets: List[Optional[nn.Module]] = list()
    ssl_auxiliary_nets: List[Optional[nn.Module]] = list()

    in_channels = 3
    image_size = 32
    block_output_dimension = 0

    mlp_kwargs = dict(output_dim=len(CLASSES), n_hidden_layers=aux_mlp_n_hidden_layers, hidden_dim=aux_mlp_hidden_dim)
    for i in range(len(config)):
        if config[i] == 'M':
            blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            auxiliary_nets.append(None)
            ssl_auxiliary_nets.append(None)
        else:
            out_channels = config[i]
            block_layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU()]
            in_channels = out_channels  # The input channels of the next convolution layer.

            if dropout_prob > 0:
                block_layers.append(nn.Dropout(dropout_prob))

            blocks.append(nn.Sequential(*block_layers))

            ssl_input_image_size = image_size
            if (i+1 < len(config)) and (config[i+1] == 'M'):
                image_size //= 2  # The auxiliary network will get the pooled output, to increase efficiency.

            block_output_dimension = out_channels * (image_size ** 2)
            auxiliary_nets.append(get_mlp(input_dim=block_output_dimension, **mlp_kwargs))

            upsampling_kwargs = dict(image_size=ssl_input_image_size, target_image_size=32) if upsample else dict()
            ssl_auxiliary_nets.append(get_ssl_aux_net(channels=out_channels, **upsampling_kwargs))

    return blocks, auxiliary_nets, ssl_auxiliary_nets, block_output_dimension


class VGG(nn.Module):
    def __init__(self,
                 vgg_name: str,
                 aux_mlp_n_hidden_layers: int = 1,
                 aux_mlp_hidden_dim: int = 1024,
                 dropout_prob: float = 0,
                 padding_mode: str = 'zeros'):
        """
        Constructs a new VGG model, that is trained in a regular end-to-end fashion.

        :param vgg_name: Should be a key in the `configs` in the head of this file.
                         This describes the architecture of the desired model.
        :param aux_mlp_n_hidden_layers: Number of hidden layers in the final MLP sub-network (predicting the scores).
        :param aux_mlp_hidden_dim: The dimension of each hidden layer in the final MLP sub-network.
        :param dropout_prob: When positive, add dropout after each non-linearity.
        :param padding_mode: Should be 'zeros' or 'circular' indicating the padding mode of each Conv layer.
        """
        super(VGG, self).__init__()
        layers, _, _, features_output_dimension = get_blocks(configs[vgg_name],
                                                             aux_mlp_n_hidden_layers,
                                                             aux_mlp_hidden_dim,
                                                             dropout_prob,
                                                             padding_mode)
        self.features = nn.Sequential(*layers)
        self.classifier = get_mlp(input_dim=features_output_dimension,
                                  output_dim=len(CLASSES),
                                  n_hidden_layers=aux_mlp_n_hidden_layers,
                                  hidden_dim=aux_mlp_hidden_dim)

    def forward(self, x: torch.Tensor):
        features = self.features(x)
        outputs = self.classifier(features)
        return outputs


class VGGwDGL(nn.Module):
    def __init__(self,
                 vgg_name,
                 aux_mlp_n_hidden_layers: int = 1,
                 aux_mlp_hidden_dim: int = 1024,
                 dropout_prob: float = 0,
                 padding_mode: str = 'zeros',
                 use_ssl: bool = False,
                 upsample: bool = False):
        """
        Constructs a new VGG model, that is trained in a local fashion (local predictions loss and possible SSL loss).

        :param vgg_name: Should be a key in the `configs` in the head of this file.
                         This describes the architecture of the desired model.
        :param aux_mlp_n_hidden_layers: Number of hidden layers in each prediction auxiliary network.
        :param aux_mlp_hidden_dim: The dimension of each hidden layer in each prediction auxiliary network.
        :param dropout_prob: When positive, add dropout after each non-linearity.
        :param padding_mode: Should be 'zeros' or 'circular' indicating the padding mode of each Conv layer.
        :param use_ssl: If true, predict the SSL outputs in addition to classes' scores.
        :param upsample: If true, the SSL outputs will be upsampled to 32 x 32.
                         Otherwise, the SSL outputs are of the same spatial size
                         as the corresponding block's output tensor.
        """
        super(VGGwDGL, self).__init__()
        blocks, auxiliary_networks, ssl_auxiliary_nets, _ = get_blocks(configs[vgg_name],
                                                                       aux_mlp_n_hidden_layers,
                                                                       aux_mlp_hidden_dim,
                                                                       dropout_prob,
                                                                       padding_mode,
                                                                       upsample)
        self.use_ssl = use_ssl
        self.blocks = nn.ModuleList(blocks)
        self.auxiliary_nets = nn.ModuleList(auxiliary_networks)
        self.ssl_auxiliary_nets = nn.ModuleList(ssl_auxiliary_nets) if use_ssl else None

    def forward(self, x: torch.Tensor, first_block_index: int = 0, last_block_index: Optional[int] = None):
        """
        Performs a forward-pass through the network, possibly using only a certain portion of the blocks.

        :param x: The input tensor to feed forward. Could be the input images,
                  as well as some intermediate hidden layer representation.
        :param first_block_index: The index of the first block to feed through.
        :param last_block_index: The index of the last block to feed through. If None - it's the last block.
        :return: The input tensor x representation (before feeding into the auxiliary networks to get predictions),
                 as well as the classes' scores predicted by the relevant auxiliary network,
                 as well as the SSL outputs predicted by the relevant SSL auxiliary network.
        """
        if last_block_index is None:
            last_block_index = len(self.blocks) - 1
            if isinstance(self.blocks[-1], nn.MaxPool2d):
                last_block_index -= 1

        representation = x
        for i in range(first_block_index, last_block_index + 1):
            representation = self.blocks[i](representation)

        next_is_pool = ((last_block_index+1 < len(self.blocks)) and
                        isinstance(self.blocks[last_block_index+1], nn.MaxPool2d))
        scores_aux_net_input = self.blocks[last_block_index+1](representation) if next_is_pool else representation

        scores_aux_net = self.auxiliary_nets[last_block_index]
        ssl_aux_net = self.ssl_auxiliary_nets[last_block_index] if (self.ssl_auxiliary_nets is not None) else None

        scores_outputs = scores_aux_net(scores_aux_net_input) if scores_aux_net is not None else None
        ssl_outputs = ssl_aux_net(representation) if scores_aux_net is not None else None

        return representation, scores_outputs, ssl_outputs
