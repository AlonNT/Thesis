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


def get_blocks(config: List[Union[int, str]],
               aux_mlp_n_hidden_layers: int = 1,
               aux_mlp_hidden_dim: int = 1024,
               dropout_prob: float = 0):
    blocks: List[nn.Sequential] = list()
    auxiliary_nets: List[nn.Sequential] = list()

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

        in_channels = out_channels
        i += 1

        if config[i] == 'M':
            block_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            image_size //= 2
            i += 1
        
        blocks.append(nn.Sequential(*block_layers))

        block_output_dimension = out_channels * (image_size ** 2)
        auxiliary_nets.append(get_mlp(input_dim=block_output_dimension, **mlp_kwargs))

    return blocks, auxiliary_nets, block_output_dimension


class VGG(nn.Module):
    def __init__(self,
                 vgg_name,
                 aux_mlp_n_hidden_layers: int = 1,
                 aux_mlp_hidden_dim: int = 1024,
                 dropout_prob: float = 0):
        super(VGG, self).__init__()
        layers, _, features_output_dimension = get_blocks(configs[vgg_name],
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
                 dropout_prob: float = 0):
        super(VGGwDGL, self).__init__()
        blocks, auxiliary_networks, _ = get_blocks(configs[vgg_name],
                                                   aux_mlp_n_hidden_layers,
                                                   aux_mlp_hidden_dim,
                                                   dropout_prob)
        self.blocks, self.auxiliary_nets = nn.ModuleList(blocks), nn.ModuleList(auxiliary_networks)

    def forward(self, x: torch.Tensor, first_block_index: int = 0, last_block_index: Optional[int] = None):
        if last_block_index is None:
            last_block_index = len(self.blocks) - 1

        for i in range(first_block_index, last_block_index + 1):
            x = self.blocks[i](x)

        outputs = self.auxiliary_nets[last_block_index](x)

        return x, outputs
