import math
from typing import List, Dict, Union, Optional

import torch

from dni import dni
from consts import CLASSES

dispatcher = {
    'conv': torch.nn.Conv2d,
    'pool': torch.nn.MaxPool2d,
    'relu': torch.nn.ReLU,
    'affine': torch.nn.Linear,
    'BatchNorm2d': torch.nn.BatchNorm2d,
    'BatchNorm1d': torch.nn.BatchNorm1d,
}


class MLPAuxiliaryNet(torch.nn.Module):
    """
    Basic `Synthesizer` based on an MLP with ReLU activation.

    Args:
        output_dim: Dimensionality of the synthesized `messages`.
        n_hidden (optional): Number of hidden layers. Defaults to 0.
        hidden_dim (optional): Dimensionality of the hidden layers. Defaults to
            `output_dim`.
        trigger_dim (optional): Dimensionality of the trigger. Defaults to
            `output_dim`.
        context_dim (optional): Dimensionality of the context. If `None`, do
            not use context. Defaults to `None`.
    """

    def __init__(self, output_dim: int, n_hidden: int = 0, hidden_dim: Optional[int] = None,
                 trigger_dim: Optional[int] = None, context_dim: Optional[int] = None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim: int = output_dim
        if trigger_dim is None:
            trigger_dim: int = output_dim

        top_layer_dim: int = output_dim if n_hidden == 0 else hidden_dim

        self.input_trigger: torch.nn.Linear = torch.nn.Linear(in_features=trigger_dim, out_features=top_layer_dim)

        if context_dim is not None:
            self.input_context: torch.nn.Linear = torch.nn.Linear(in_features=context_dim, out_features=top_layer_dim)
        else:
            self.input_context = None

        self.layers: torch.nn.ModuleList = torch.nn.ModuleList([
            torch.nn.Linear(in_features=hidden_dim,
                            out_features=(hidden_dim if layer_index < n_hidden - 1 else output_dim))
            for layer_index in range(n_hidden)
        ])

        # zero-initialize the last layer, as in the paper
        last_layer: torch.nn.Linear = self.layers[-1] if n_hidden > 0 else self.input_trigger
        torch.nn.init.constant_(last_layer.weight, 0)
        if (n_hidden == 0) and (context_dim is not None):
            torch.nn.init.constant_(self.input_context.weight, 0)

    def forward(self, trigger, context):
        """Synthesizes a `message` based on `trigger` and `context`.

        Args:
            trigger: `trigger` to synthesize the `message` based on. Size:
                (`batch_size`, `trigger_dim`).
            context: `context` to condition the synthesizer. Ignored if
                `context_dim` has not been specified in the constructor. Size:
                (`batch_size`, `context_dim`).

        Returns:
            The synthesized `message`.
        """
        x = self.input_trigger(trigger)

        if self.input_context is not None:
            x += self.input_context(context)

        for layer in self.layers:
            x = layer(torch.nn.functional.relu(x))

        return x


class ConvAuxiliaryNet(torch.nn.Module):
    def __init__(self, n_channels: int, context_dim: int = len(CLASSES)):
        super(ConvAuxiliaryNet, self).__init__()

        self.input_trigger: torch.nn.Module = torch.nn.Conv2d(n_channels, n_channels, kernel_size=5, padding=2)
        self.input_trigger_bn: torch.nn.Module = torch.nn.BatchNorm2d(n_channels)
        self.input_context: torch.nn.Module = torch.nn.Linear(context_dim, n_channels)
        self.hidden: torch.nn.Module = torch.nn.Conv2d(n_channels, n_channels, kernel_size=5, padding=2)
        self.hidden_bn: torch.nn.Module = torch.nn.BatchNorm2d(n_channels)
        self.output: torch.nn.Module = torch.nn.Conv2d(n_channels, n_channels, kernel_size=5, padding=2)

        # zero-initialize the last layer, as in the paper
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
    def __init__(self,
                 architecture: List[Dict[str, Union[str, Dict[str, int]]]],
                 dni_positions: Union[None, str, List[int]]):
        super(MainNetDNI, self).__init__()

        self.layers: torch.nn.ModuleList = torch.nn.ModuleList(
            [dispatcher[layer['type']](**layer['args']) for layer in architecture]
        )

        self.dni_aux_nets: torch.nn.ModuleDict = self.get_auxiliary_nets(dni_positions)

    def forward(self, x):
        input_flattened = False
        for i, layer in enumerate(self.layers):
            # Flatten the convolutional layer's output to feed the linear layer.
            if isinstance(layer, torch.nn.Linear) and not input_flattened:
                x = x.view(-1, math.prod(x.shape[1:]))
                input_flattened = True

            x = layer(x)

            if str(i) in self.dni_aux_nets:
                x = self.dni_aux_nets[str(i)](x)  # TODO enable using context

        return x

    def get_positions(self, positions: List[int]) -> List[int]:
        if positions is None:
            return list()
        elif positions == 'every_layer':
            return [i for i, l in enumerate(self.layers)
                    if isinstance(l, torch.nn.Conv2d) or isinstance(l, torch.nn.Linear)]
        else:
            assert isinstance(positions, list) and len(positions) > 0 and all(isinstance(i, int) for i in positions), \
                "positions should be None, string or a list of indices"
            return positions

    def get_auxiliary_nets(self, positions: Union[None, str, List[int]]):
        positions: List[int] = self.get_positions(positions)

        auxiliary_nets: Dict[int, Union[torch.nn.Module]] = dict()
        for i in positions:
            layer: torch.nn.Module = self.layers[i]

            if isinstance(layer, torch.nn.Conv2d):
                aux_net = ConvAuxiliaryNet(n_channels=layer.out_channels)
            elif isinstance(layer, torch.nn.Linear):
                aux_net = MLPAuxiliaryNet(output_dim=layer.out_features)
            else:
                raise ValueError(f'Auxiliary network was asked for layer {i} but it\'s not linear/convolutional layer.')

            auxiliary_nets[i] = dni.BackwardInterface(aux_net)

        # For some reason `ModuleDict` expects its keys to be strings, so be it.
        return torch.nn.ModuleDict({str(k): v for k, v in auxiliary_nets.items()})


def get_mlp(input_dim: int, output_dim: int, n_hidden_layers: int = 0, hidden_dim: int = 0) -> torch.nn.Sequential:
    layers = [torch.nn.Flatten()]

    for i in range(n_hidden_layers):
        in_features = input_dim if i == 0 else hidden_dim
        layers.extend([torch.nn.Linear(in_features=in_features, out_features=hidden_dim), torch.nn.ReLU()])

    output_layer_input_dim = input_dim if n_hidden_layers == 0 else hidden_dim
    layers.append(torch.nn.Linear(in_features=output_layer_input_dim, out_features=output_dim))

    return torch.nn.Sequential(*layers)


def get_cnn(conv_layers_channels=None, affine_layers_channels=None):
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
            layers.append(torch.nn.ReLU())

        in_features = affine_layer_channels

    return torch.nn.Sequential(*layers)


class MainNetDGL(torch.nn.Module):
    def __init__(self):
        super(MainNetDGL, self).__init__()

        self.conv1_channels = 16
        self.conv2_channels = 32
        self.conv3_channels = 64
        self.conv_kernel_size = 5
        self.conv_padding = 2
        self.pool_kernel_size = 2
        image_size = 32

        self.blocks: torch.nn.ModuleList = torch.nn.ModuleList([])
        self.auxiliary_nets: torch.nn.ModuleList = torch.nn.ModuleList([])

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

        self.aux_net_hidden_dim = 256
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

        return inputs_representation, outputs
