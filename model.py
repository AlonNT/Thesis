import torch

from typing import List, Optional

from dni import dni
from consts import CLASSES


def get_mlp(input_dim: int, output_dim: int, n_hidden_layers: int = 0, hidden_dim: int = 0) -> torch.nn.Sequential:
    """
    This function builds a MLP (i.e. Multi-Layer-Perceptron) and return it as a PyTorch's sequential model.
    :param input_dim: The dimension of the input tensor.
    :param output_dim: The dimension of the output tensor.
    :param n_hidden_layers: Number of hidden layers.
    :param hidden_dim: The dimension of each hidden layer.
    :return: A sequential model which is the constructed MLP.
    """
    layers = [torch.nn.Flatten()]  # Begins with a flatten layer (useful when the input is 4D from a conv layer).

    for i in range(n_hidden_layers):
        layers.append(torch.nn.Linear(in_features=input_dim if i == 0 else hidden_dim,
                                      out_features=hidden_dim))
        layers.append(torch.nn.ReLU())

    layers.append(torch.nn.Linear(in_features=input_dim if n_hidden_layers == 0 else hidden_dim,
                                  out_features=output_dim))

    return torch.nn.Sequential(*layers)


def get_cnn(conv_layers_channels: Optional[List[int]] = None,
            affine_layers_channels: Optional[List[int]] = None) -> torch.nn.Sequential:
    """
    This function builds a CNN and return it as a PyTorch's sequential model.
    :param conv_layers_channels: A list of integers containing the channels of each convolution block.
                                 Each block will contain Conv-BatchNorm-MaxPool-ReLU.
    :param affine_layers_channels: A list of integers containing the channels of each linear layer.
    :return: A sequential model which is the constructed CNN.
    """
    if conv_layers_channels is None:
        conv_layers_channels = [16, 32, 64]
    if affine_layers_channels is None:
        affine_layers_channels = [256, len(CLASSES)]

    conv_kernel_size = 5
    padding = 2
    pool_kernel_size = 2
    image_size = 32

    layers = list()

    in_channels = 3  # 3 channels corresponding to RGB channels in the original images.
    for conv_layer_channels in conv_layers_channels:
        out_channels = conv_layer_channels

        layers.append(torch.nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=conv_kernel_size,
                                      padding=padding))
        layers.append(torch.nn.BatchNorm2d(out_channels))
        layers.append(torch.nn.MaxPool2d(kernel_size=pool_kernel_size))
        layers.append(torch.nn.ReLU())

        in_channels = out_channels

    layers.append(torch.nn.Flatten())

    down_sample_factor = 2 ** len(conv_layers_channels)
    spatial_size = image_size // down_sample_factor
    in_features = conv_layers_channels[-1] * (spatial_size ** 2)
    for i, affine_layer_channels in enumerate(affine_layers_channels):
        layers.append(torch.nn.Linear(in_features=in_features, out_features=affine_layer_channels))
        if i < len(affine_layers_channels) - 1:
            layers.append(torch.nn.ReLU())  # Do not append ReLU in the last affine layer.

        in_features = affine_layer_channels

    return torch.nn.Sequential(*layers)


class ConvSynthesizer(torch.nn.Module):
    """
    This class is a convolutional synthesizer for creating synthetic gradients
    to incorporate in a decoupled neural interface.
    """
    def __init__(self, n_channels: int, context_dim: int = len(CLASSES)):
        super(ConvSynthesizer, self).__init__()

        self.input_trigger = torch.nn.Conv2d(n_channels, n_channels, kernel_size=5, padding=2)
        self.input_trigger_bn = torch.nn.BatchNorm2d(n_channels)
        self.input_context = torch.nn.Linear(context_dim, n_channels)
        self.hidden = torch.nn.Conv2d(n_channels, n_channels, kernel_size=5, padding=2)
        self.hidden_bn = torch.nn.BatchNorm2d(n_channels)
        self.output = torch.nn.Conv2d(n_channels, n_channels, kernel_size=5, padding=2)

        # Zero-initialize the last layer, as in the DNI paper.
        torch.nn.init.constant_(self.output.weight, 0)

    def forward(self, trigger, context):
        x = self.input_trigger(trigger)

        if context is not None:
            x += self.input_context(context).unsqueeze(2).unsqueeze(3).expand_as(x)

        x = self.input_trigger_bn(x)
        x = torch.nn.functional.relu(x)
        x = self.hidden(x)
        x = self.hidden_bn(x)
        x = torch.nn.functional.relu(x)
        x = self.output(x)

        return x


class MainNetDNI(torch.nn.Module):
    """
    This class represents the main network which uses Decoupled-Neural-Interfaces.
    """
    def __init__(self):
        super(MainNetDNI, self).__init__()

        # Hyper-parameters defining the architecture.
        self.conv1_channels = 16
        self.conv2_channels = 32
        self.conv3_channels = 64
        self.conv_kernel_size = 5
        self.conv_padding = 2
        self.pool_kernel_size = 2
        self.mlp_hidden_dim = 256
        self.mlp_n_hidden_layers = 1
        image_size = 32

        # The network begins with 'blocks', each is a Conv - BatchNorm - MaxPool - ReLU
        self.blocks: torch.nn.ModuleList = torch.nn.ModuleList([])

        self.blocks.append(torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=self.conv1_channels,
                            kernel_size=self.conv_kernel_size,
                            padding=self.conv_padding),
            torch.nn.BatchNorm2d(self.conv1_channels),
            torch.nn.MaxPool2d(kernel_size=self.pool_kernel_size),
            torch.nn.ReLU(),
        ))

        self.blocks.append(torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.conv1_channels,
                            out_channels=self.conv2_channels,
                            kernel_size=self.conv_kernel_size,
                            padding=self.conv_padding),
            torch.nn.BatchNorm2d(self.conv2_channels),
            torch.nn.MaxPool2d(kernel_size=self.pool_kernel_size),
            torch.nn.ReLU(),
        ))

        self.blocks.append(torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.conv2_channels,
                            out_channels=self.conv3_channels,
                            kernel_size=self.conv_kernel_size,
                            padding=self.conv_padding),
            torch.nn.BatchNorm2d(self.conv3_channels),
            torch.nn.MaxPool2d(kernel_size=self.pool_kernel_size),
            torch.nn.ReLU()
        ))

        # The final component if the MLP (Multi-Layer-Perceptron) which is a network
        # containing several linear layers with ReLU activations in between.
        down_sample_factor = 2 ** len(self.blocks)             # How many down-samples were done in the blocks.
        spatial_size = image_size // down_sample_factor        # The spatial size of the tensor -  its height/width.
        input_dim = self.conv3_channels * (spatial_size ** 2)  # The total dimension which is the input for the mlp.
        self.mlp = get_mlp(input_dim=input_dim, output_dim=len(CLASSES),
                           n_hidden_layers=self.mlp_n_hidden_layers, hidden_dim=self.mlp_hidden_dim)

        # The backward interfaces decouple each block from its downstream blocks.
        self.backward_interfaces = torch.nn.ModuleList([
            dni.BackwardInterface(ConvSynthesizer(n_channels))
            for n_channels in [self.conv1_channels, self.conv2_channels, self.conv3_channels]
        ])

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
            x = self.backward_interfaces[i](x)

        classes_scores = self.mlp(x)

        return classes_scores


class MainNetDGL(torch.nn.Module):
    """
    This class represents the main network which uses Decoupled-Greedy-Learning.
    """
    def __init__(self):
        super(MainNetDGL, self).__init__()

        # Hyper-parameters defining the architecture.
        self.conv1_channels = 16
        self.conv2_channels = 32
        self.conv3_channels = 64
        self.conv_kernel_size = 5
        self.conv_padding = 2
        self.pool_kernel_size = 2
        self.aux_net_hidden_dim = 256
        image_size = 32

        # The network is build of 'blocks', each is a Conv - BatchNorm - MaxPool - ReLU.
        self.blocks: torch.nn.ModuleList = torch.nn.ModuleList([])

        self.blocks.append(torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=self.conv1_channels,
                            kernel_size=self.conv_kernel_size,
                            padding=self.conv_padding),
            torch.nn.BatchNorm2d(self.conv1_channels),
            torch.nn.MaxPool2d(kernel_size=self.pool_kernel_size),
            torch.nn.ReLU(),
        ))

        self.blocks.append(torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.conv1_channels,
                            out_channels=self.conv2_channels,
                            kernel_size=self.conv_kernel_size,
                            padding=self.conv_padding),
            torch.nn.BatchNorm2d(self.conv2_channels),
            torch.nn.MaxPool2d(kernel_size=self.pool_kernel_size),
            torch.nn.ReLU(),
        ))

        self.blocks.append(torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.conv2_channels,
                            out_channels=self.conv3_channels,
                            kernel_size=self.conv_kernel_size,
                            padding=self.conv_padding),
            torch.nn.BatchNorm2d(self.conv3_channels),
            torch.nn.MaxPool2d(kernel_size=self.pool_kernel_size),
            torch.nn.ReLU()
        ))

        # The auxiliary networks are used to generate gradients to update the blocks of the network.
        # Each auxiliary network is used to generate gradients to the corresponding block.
        # Note that the last auxiliary network is a part of the final computational graph.
        self.auxiliary_nets: torch.nn.ModuleList = torch.nn.ModuleList([])
        mlp_kwargs = dict(output_dim=len(CLASSES), n_hidden_layers=1, hidden_dim=self.aux_net_hidden_dim)
        self.auxiliary_nets.append(get_mlp(input_dim=self.conv1_channels * ((image_size // 2) ** 2), **mlp_kwargs))
        self.auxiliary_nets.append(get_mlp(input_dim=self.conv2_channels * ((image_size // 4) ** 2), **mlp_kwargs))
        self.auxiliary_nets.append(get_mlp(input_dim=self.conv3_channels * ((image_size // 8) ** 2), **mlp_kwargs))

    def forward(self, x, first_block_index=0, last_block_index=None):
        if last_block_index is None:
            last_block_index = len(self.blocks) - 1

        for i in range(first_block_index, last_block_index + 1):
            x = self.blocks[i](x)

        inputs_representation = x
        outputs = self.auxiliary_nets[last_block_index](inputs_representation)

        # In addition to the outputs, return the input representation (i.e. the output of the last layer of the
        # actual graph, before it's being fed to the corresponding auxiliary network).
        return inputs_representation, outputs
