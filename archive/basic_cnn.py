import torch

from typing import Optional

from dni import dni
from consts import CLASSES
from utils import one_hot, get_mlp


class ConvSynthesizer(torch.nn.Module):
    """
    This class is a convolutional synthesizer for creating synthetic gradients
    to incorporate in a decoupled neural interface.
    It consists of Conv - BatchNorm - ReLU - Conv - BatchNorm - ReLU - Conv
    where all convolutional layers contain the same number of channels (i.e. the `n_channels` argument).
    If DNI with context is being used, the context vector (which is a one-hot vector describing the label) is
    being multiplied by an affine layer and added to the first convolutional layer.
    """
    def __init__(self, n_channels: int, context_dim: int = len(CLASSES)):
        super(ConvSynthesizer, self).__init__()

        self.input_trigger = torch.nn.Conv2d(n_channels, n_channels, kernel_size=5, padding=2)
        self.input_trigger_bn = torch.nn.BatchNorm2d(n_channels)
        self.input_context = torch.nn.Linear(context_dim, n_channels)
        self.hidden = torch.nn.Conv2d(n_channels, n_channels, kernel_size=5, padding=2)
        self.hidden_bn = torch.nn.BatchNorm2d(n_channels)
        self.output = torch.nn.Conv2d(n_channels, n_channels, kernel_size=5, padding=2)

        torch.nn.init.constant_(self.output.weight, 0)  # Zero-initialize the last layer, as in the DNI paper.

    def forward(self, trigger, context):
        """
        Perform a forward pass through the ConvSynthesizer model.

        :param trigger: The input "trigger" (terminology as in the DNI paper).
                        Essentially, this is the output tensor of the layer
                        which we want to "answer" with the synthetic gradient with respect to this output tensor.
        :param context: The context vector, which is the label corresponding to the input trigger.
        :return: The ConvSynthesizer's output, which is the synthetic gradient with respect to the input trigger.
        """
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


class CNNwDNI(torch.nn.Module):
    """
    This class represents the main network which uses Decoupled-Neural-Interfaces.
    The network consists of 3 blocks of Conv - BatchNorm - MaxPool - ReLU.
    During training, each block will receive its feedback (i.e. downstream gradients)
    from the synthetic gradients module, which is a ConvSynthesizer (see above).
    """
    def __init__(self, use_context: bool = False):
        super(CNNwDNI, self).__init__()

        self.use_context = use_context

        # Hyper-parameters defining the architecture, same as DGL & DNI papers.
        self.conv1_channels = 128
        self.conv2_channels = 128
        self.conv3_channels = 128
        self.conv_kernel_size = 5
        self.conv_padding = 2
        self.pool_kernel_size = 2  # not the same as DNI & DGL comparison in DGL paper (there it was 3).
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
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=self.pool_kernel_size)
        ))
        self.blocks.append(torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.conv1_channels,
                            out_channels=self.conv2_channels,
                            kernel_size=self.conv_kernel_size,
                            padding=self.conv_padding),
            torch.nn.BatchNorm2d(self.conv2_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=self.pool_kernel_size)  # not the same as DNI & DGL comparison in DGL paper.
        ))
        self.blocks.append(torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.conv2_channels,
                            out_channels=self.conv3_channels,
                            kernel_size=self.conv_kernel_size,
                            padding=self.conv_padding),
            torch.nn.BatchNorm2d(self.conv3_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=self.pool_kernel_size)  # not the same as DNI & DGL comparison in DGL paper.
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

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform a forward pass through the model.

        :param x: The input tensor (i.e. the images).
        :param y: The labels, will be used as context for DNI with context.
                  Defaults to None which means no context will be used.
        :return: The predictions (i.e. classes' scores) of the model.
        """
        for i, block in enumerate(self.blocks):
            x = block(x)

            if self.training:
                if self.blocks[0][0].weight.is_cuda:
                    device = torch.device(f'cuda:{self.blocks[0][0].weight.get_device()}')
                else:
                    device = torch.device('cpu')
                context = one_hot(y, device) if self.use_context else None
                with dni.synthesizer_context(context):
                    x = self.backward_interfaces[i](x)

        classes_scores = self.mlp(x)

        return classes_scores


class CNNwDGL(torch.nn.Module):
    """
    This class represents the main network which is a CNN using Decoupled-Greedy-Learning.
    The network consists of 3 blocks of Conv - BatchNorm - MaxPool - ReLU.
    After each block there is an auxiliary network which is a multi-layer-perceptron (MLP).
    The last MLP is conceptually a part of the network, since in test-mode it will provide the predictions.
    """
    def __init__(self):
        super(CNNwDGL, self).__init__()

        # Hyper-parameters defining the architecture.
        self.conv1_channels = 128
        self.conv2_channels = 128
        self.conv3_channels = 128
        self.conv_kernel_size = 5
        self.conv_padding = 2
        self.pool_kernel_size = 2
        self.aux_net_hidden_dim = 256
        self.aux_net_n_hidden_layers = 1
        image_size = 32

        # The network is build of 'blocks', each is a Conv - BatchNorm - MaxPool - ReLU.
        self.blocks: torch.nn.ModuleList = torch.nn.ModuleList([])
        self.blocks.append(torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=self.conv1_channels,
                            kernel_size=self.conv_kernel_size,
                            padding=self.conv_padding),
            torch.nn.BatchNorm2d(self.conv1_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=self.pool_kernel_size)
        ))
        self.blocks.append(torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.conv1_channels,
                            out_channels=self.conv2_channels,
                            kernel_size=self.conv_kernel_size,
                            padding=self.conv_padding),
            torch.nn.BatchNorm2d(self.conv2_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=self.pool_kernel_size)
        ))
        self.blocks.append(torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.conv2_channels,
                            out_channels=self.conv3_channels,
                            kernel_size=self.conv_kernel_size,
                            padding=self.conv_padding),
            torch.nn.BatchNorm2d(self.conv3_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=self.pool_kernel_size)
        ))

        # The auxiliary networks are used to generate feedback to update the blocks of the network.
        # Each auxiliary network is used to generate gradients to the corresponding block.
        # Note that the last auxiliary network is a part of the final computational graph.
        self.auxiliary_nets: torch.nn.ModuleList = torch.nn.ModuleList([])
        mlp_kwargs = dict(output_dim=len(CLASSES),
                          n_hidden_layers=self.aux_net_n_hidden_layers, hidden_dim=self.aux_net_hidden_dim)
        self.auxiliary_nets.append(get_mlp(input_dim=self.conv1_channels * ((image_size // 2) ** 2), **mlp_kwargs))
        self.auxiliary_nets.append(get_mlp(input_dim=self.conv2_channels * ((image_size // 4) ** 2), **mlp_kwargs))
        self.auxiliary_nets.append(get_mlp(input_dim=self.conv3_channels * ((image_size // 8) ** 2), **mlp_kwargs))

    def forward(self, x: torch.Tensor, first_block_index: int = 0, last_block_index: Optional[int] = None):
        """
        Perform a forward pass through the model.

        :param x: The input tensor. It can be the actual input (i.e. the images themselves)
                  or some intermediate representation (i.e. the output of some intermediate previous layer).
        :param first_block_index: The first block in the model to feed with the input tensor `x`.
                                  Default is 0 which means the first block of the network.
        :param last_block_index: The last block in the model to use, which its output will be returned.
                                 Default is the last block of the network.
        :return: A tuple containing two elements:
                 (1) The input `x` representation (i.e. the output of the block indicated by `last_block_index`,
                     before it's being fed to the relevant auxiliary network to get the classes scores predictions).
                 (2) The predictions (i.e. classes' scores) of the auxiliary network corresponding to the given
                     `last_block_index`, when it's the last block it's the last auxiliary network
                     which is conceptually a part of the model.
        """
        if last_block_index is None:
            last_block_index = len(self.blocks) - 1

        for i in range(first_block_index, last_block_index + 1):
            x = self.blocks[i](x)

        outputs = self.auxiliary_nets[last_block_index](x)

        # In addition to the outputs, return the input representation (i.e. the output of the last layer of the
        # actual graph, before it's being fed to the corresponding auxiliary network).
        return x, outputs
