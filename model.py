import math
from typing import List, Dict, Union, Optional

import torch

from dni import dni
from consts import CLASSES


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

    def __init__(self, output_dim: int, n_hidden: int = 0,
                 hidden_dim: Optional[int] = None, trigger_dim: Optional[int] = None, context_dim: Optional[int] = None,
                 predict_classes_scores: bool = False):
        super().__init__()

        if hidden_dim is None:
            hidden_dim: int = output_dim
        if trigger_dim is None:
            trigger_dim: int = output_dim

        top_layer_dim: int = output_dim if n_hidden == 0 else hidden_dim

        self.predict_classes_scores: bool = predict_classes_scores
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

        if predict_classes_scores:
            self.scores: torch.nn.Module = torch.nn.Linear(in_features=last_layer.out_features,
                                                           out_features=len(CLASSES))

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

        if self.predict_classes_scores:
            x = self.scores(x)

        return x


class ConvAuxiliaryNet(torch.nn.Module):
    def __init__(self, n_channels: int, spatial_size: int,
                 context_dim: int = len(CLASSES), predict_classes_scores: bool = False):
        super(ConvAuxiliaryNet, self).__init__()

        self.predict_classes_scores: bool = predict_classes_scores

        self.input_trigger: torch.nn.Module = torch.nn.Conv2d(n_channels, n_channels, kernel_size=5, padding=2)
        self.input_trigger_bn: torch.nn.Module = torch.nn.BatchNorm2d(n_channels)
        self.input_context: torch.nn.Module = torch.nn.Linear(context_dim, n_channels)
        self.hidden: torch.nn.Module = torch.nn.Conv2d(n_channels, n_channels, kernel_size=5, padding=2)
        self.hidden_bn: torch.nn.Module = torch.nn.BatchNorm2d(n_channels)
        self.output: torch.nn.Module = torch.nn.Conv2d(n_channels, n_channels, kernel_size=5, padding=2)

        if predict_classes_scores:
            self.scores: torch.nn.Module = torch.nn.Linear(in_features=(n_channels * (spatial_size ** 2)),
                                                           out_features=len(CLASSES))

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

        if self.predict_classes_scores:
            x = self.scores(x)

        return x


class MainNet(torch.nn.Module):
    dispatcher = {
        'conv': torch.nn.Conv2d,
        'pool': torch.nn.MaxPool2d,
        'relu': torch.nn.ReLU,
        'affine': torch.nn.Linear,
        'BatchNorm2d': torch.nn.BatchNorm2d,
        'BatchNorm1d': torch.nn.BatchNorm1d,
    }

    def __init__(self,
                 architecture: List[Dict[str, Union[str, Dict[str, int]]]],
                 dni_positions: Union[None, str, List[int]],
                 dgl_positions: Union[None, str, List[int]]):
        super(MainNet, self).__init__()
        self.layers: torch.nn.ModuleList = torch.nn.ModuleList(
            [MainNet.dispatcher[layer['type']](**layer['args']) for layer in architecture]
        )

        self.dni_aux_nets: torch.nn.ModuleDict = self.get_auxiliary_nets(dni_positions, is_dni=True)
        self.dgl_aux_nets: torch.nn.ModuleDict = self.get_auxiliary_nets(dgl_positions, is_dni=False)

    def forward(self, x):
        input_flattened = False
        for i, layer in enumerate(self.layers):
            # Flatten the convolutional layer's output to feed the linear layer.
            if isinstance(layer, torch.nn.Linear) and not input_flattened:
                x = x.view(-1, math.prod(x.shape[1:]))
                input_flattened = True

            x = layer(x)

            if str(i) in self.dni_aux_nets:
                # TODO context
                x = self.dni_aux_nets[str(i)](x)

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

    def get_auxiliary_nets(self, positions: Union[None, str, List[int]], is_dni: bool):
        positions: List[int] = self.get_positions(positions)

        auxiliary_nets: Dict[int, Union[torch.nn.Module]] = dict()
        image_size: int = 32  # This is the height and width of the images in CIFAR10 dataset.
        for i in positions:
            layer: torch.nn.Module = self.layers[i]

            if isinstance(layer, torch.nn.Conv2d):
                n_down_sampling: int = sum(isinstance(self.layers[j], torch.nn.MaxPool2d) for j in range(i))
                aux_net = ConvAuxiliaryNet(
                    n_channels=layer.out_channels,
                    spatial_size=image_size * (1 / (2 ** n_down_sampling)),
                    predict_classes_scores=not is_dni
                )
            elif isinstance(layer, torch.nn.Linear):
                aux_net = MLPAuxiliaryNet(output_dim=layer.out_features, predict_classes_scores=not is_dni)
            else:
                raise ValueError(f'Auxiliary network was asked for layer {i} but it\'s not linear/convolutional layer.')

            auxiliary_nets[i] = dni.BackwardInterface(aux_net) if is_dni else aux_net

        # For some reason `ModuleDict` expects its keys to be strings, so be it.
        return torch.nn.ModuleDict({str(k): v for k, v in auxiliary_nets.items()})
