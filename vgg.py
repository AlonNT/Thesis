import torch
import torch.nn as nn

from typing import List, Optional, Union

from consts import CLASSES
from utils import get_mlp

# Inspiration was taken from
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

# Configurations for the VGG models family.
# A number indicates number of channels in a convolution block, and M denotes a MaxPool layer.
configs = {
    # https://github.com/anokland/local-loss/blob/master/train.py#L1276
    'VGG8b': [128, 256, 'M', 256, 512, 'M', 512, 'M', 512, 'M'],  #
    'VGG11b': [128, 128, 128, 256, 'M', 256, 512, 'M', 512, 512, 'M', 512, 'M'],

    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def get_ssl_aux_net(channels: int, image_size: Optional[int] = None, target_image_size: Optional[int] = None):
    layers = list()

    assert (image_size is None) == (target_image_size is None), \
        "image_size and target_image_size should be both None (in which case no up-sampling is done), " \
        "or both not None (in which case upsampling is done from image_size to target_image_size)."

    in_channels = channels
    if image_size is not None:  # up-sample from image_size to target_image_size.
        while image_size != target_image_size:
            out_channels = in_channels // 2
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            image_size *= 2
            in_channels = out_channels

    # Convolutional layer with 1x1 kernel to change channels to 3.
    layers.append(nn.Conv2d(in_channels, out_channels=3, kernel_size=1, padding=0))

    return nn.Sequential(*layers)


def get_blocks(config: List[Union[int, str]],
               aux_mlp_n_hidden_layers: int = 1,
               aux_mlp_hidden_dim: int = 1024,
               dropout_prob: float = 0,
               upsample: bool = False):
    blocks: List[nn.Module] = list()
    auxiliary_nets: List[nn.Module] = list()
    ssl_auxiliary_nets: List[nn.Module] = list()

    in_channels = 3
    image_size = 32
    block_output_dimension = None

    mlp_kwargs = dict(output_dim=len(CLASSES), n_hidden_layers=aux_mlp_n_hidden_layers, hidden_dim=aux_mlp_hidden_dim)
    i = 0
    while i < len(config):
        out_channels = config[i]

        block_layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU()]

        if dropout_prob > 0:
            block_layers.append(nn.Dropout(dropout_prob))

        in_channels = out_channels  # This will be the input channels of the next convolutional layer.
        i += 1

        if config[i] == 'M':
            block_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            image_size //= 2
            i += 1

        blocks.append(nn.Sequential(*block_layers))

        block_output_dimension = out_channels * (image_size ** 2)
        auxiliary_nets.append(get_mlp(input_dim=block_output_dimension, **mlp_kwargs))

        ssl_aux_net_kwargs = dict(channels=out_channels)
        if upsample:
            ssl_aux_net_kwargs.update(dict(image_size=image_size, target_image_size=32))
        ssl_auxiliary_nets.append(get_ssl_aux_net(**ssl_aux_net_kwargs))

    return blocks, auxiliary_nets, ssl_auxiliary_nets, block_output_dimension


class VGG(nn.Module):
    def __init__(self,
                 vgg_name,
                 aux_mlp_n_hidden_layers: int = 1,
                 aux_mlp_hidden_dim: int = 1024,
                 dropout_prob: float = 0):
        super(VGG, self).__init__()
        layers, _, _, features_output_dimension = get_blocks(configs[vgg_name],
                                                             aux_mlp_n_hidden_layers,
                                                             aux_mlp_hidden_dim,
                                                             dropout_prob)
        self.features = nn.Sequential(*layers)
        self.classifier = get_mlp(input_dim=features_output_dimension,
                                  output_dim=len(CLASSES),
                                  n_hidden_layers=1,
                                  hidden_dim=1024)

    def forward(self, x):
        features = self.features(x)
        outputs = self.classifier(features)
        return outputs


class VGGwDGL(nn.Module):
    def __init__(self,
                 vgg_name,
                 aux_mlp_n_hidden_layers: int = 1,
                 aux_mlp_hidden_dim: int = 1024,
                 dropout_prob: float = 0,
                 use_ssl: bool = False,
                 upsample: bool = False):
        super(VGGwDGL, self).__init__()
        blocks, auxiliary_networks, ssl_auxiliary_nets, _ = get_blocks(configs[vgg_name],
                                                                       aux_mlp_n_hidden_layers,
                                                                       aux_mlp_hidden_dim,
                                                                       dropout_prob,
                                                                       upsample)
        self.use_ssl = use_ssl
        self.blocks = nn.ModuleList(blocks)
        self.auxiliary_nets = nn.ModuleList(auxiliary_networks)
        self.ssl_auxiliary_nets = nn.ModuleList(ssl_auxiliary_nets) if use_ssl else None

    def forward(self, x: torch.Tensor, first_block_index: int = 0, last_block_index: Optional[int] = None):
        if last_block_index is None:
            last_block_index = len(self.blocks) - 1

        representation = x

        for i in range(first_block_index, last_block_index + 1):
            representation = self.blocks[i](representation)

        outputs = self.auxiliary_nets[last_block_index](representation)
        ssl_outputs = self.ssl_auxiliary_nets[last_block_index](representation) if self.use_ssl else None

        return representation, outputs, ssl_outputs
