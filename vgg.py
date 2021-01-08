import torch

from typing import List, Optional, Union

from consts import CLASSES
from utils import get_mlp

# Inspiration was taken from
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

# Configurations for the VGG models family.
# A number indicates number of channels in a convolutional block, and M denotes a MaxPool layer.
configs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def get_blocks(config: List[Union[int, str]], mlp_n_hidden_layers: int = 1, mlp_hidden_dim: int = 512):
    layers = list()
    auxiliary_nets = list()
    in_channels = 3
    image_size = 32
    mlp_kwargs = dict(output_dim=len(CLASSES), n_hidden_layers=mlp_n_hidden_layers, hidden_dim=mlp_hidden_dim)
    for i, n_channels in enumerate(config):
        if n_channels == 'M':
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            auxiliary_nets += [None]
            image_size //= 2
        else:
            layers += [torch.nn.Sequential(torch.nn.Conv2d(in_channels, n_channels, kernel_size=3, padding=1),
                                           torch.nn.BatchNorm2d(n_channels),
                                           torch.nn.ReLU())]
            in_channels = n_channels

            layer_output_dimension = n_channels * (image_size ** 2)
            if i < len(config) - 1:
                auxiliary_nets += [get_mlp(input_dim=layer_output_dimension, **mlp_kwargs)]
            else:  # In the last iteration the auxiliary network is different - it's part of the actual model
                auxiliary_nets += [torch.nn.Sequential(torch.nn.AvgPool2d(kernel_size=1, stride=1),
                                                       torch.nn.Linear(layer_output_dimension, len(CLASSES)))]
    layers += [torch.nn.AvgPool2d(kernel_size=1, stride=1)]
    auxiliary_nets += [None]
    return layers, auxiliary_nets


class VGG(torch.nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        layers, _ = get_blocks(configs[vgg_name])
        self.features = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class VGGwDGL(torch.nn.Module):
    def __init__(self, vgg_name):
        super(VGGwDGL, self).__init__()
        blocks, auxiliary_networks = get_blocks(configs[vgg_name])

        # Remove the last item from each list, because it corresponds to the AvgPool layer which is already
        # in the last block's auxiliary network.
        blocks.pop()
        auxiliary_networks.pop()

        self.blocks, self.auxiliary_nets = torch.nn.ModuleList(blocks), torch.nn.ModuleList(auxiliary_networks)

    def forward(self, x: torch.Tensor, first_block_index: int = 0, last_block_index: Optional[int] = None):
        if last_block_index is None:
            last_block_index = len(self.blocks) - 1

        for i in range(first_block_index, last_block_index + 1):
            x = self.blocks[i](x)

        aux_net = self.auxiliary_nets[last_block_index]

        outputs = aux_net(x) if aux_net is not None else None

        return x, outputs
