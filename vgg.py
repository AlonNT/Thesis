import torch
import torch.nn as nn
from typing import List, Union, Tuple, Optional

from consts import CLASSES
from utils import get_mlp, get_cnn, ShuffleTensor

# Configurations for the VGG models family.
# A number indicates number of channels in a convolution block, and M/A denotes a MaxPool/AvgPool layer.
configs = {
    'VGGc16d1': [16],
    'VGGc32d1': [32],
    'VGGc64d1': [64],
    'VGGc128d1': [128],
    'VGGc256d1': [256],
    'VGGc512d1': [512],
    'VGGc1024d1': [1024],
    'VGGc1024d1A': [1024, 'A'],

    'VGGc16d2': [16, 16],
    'VGGc32d2': [32, 32],
    'VGGc64d2': [64, 64],
    'VGGc128d2': [128, 128],
    'VGGc256d2': [256, 256],
    'VGGc512d2': [512, 512],
    'VGGc1024d2': [1024, 1024],
    'VGGc1024d2A': [1024, 'A', 1024],

    'VGGc16d3': [16, 16, 16],
    'VGGc32d3': [32, 32, 32],
    'VGGc64d3': [64, 64, 64],
    'VGGc128d3': [128, 128, 128],
    'VGGc256d3': [256, 256, 256],
    'VGGc512d3': [512, 512, 512],
    'VGGc1024d3': [1024, 1024, 1024],
    'VGGc1024d3A': [1024, 'A', 1024, 1024],

    'VGGc16d4': [16, 16, 16, 16],
    'VGGc32d4': [32, 32, 32, 32],
    'VGGc64d4': [64, 64, 64, 64],
    'VGGc128d4': [128, 128, 128, 128],
    'VGGc256d4': [256, 256, 256, 256],
    'VGGc512d4': [512, 512, 512, 512],
    'VGGc1024d4': [1024, 1024, 1024, 1024],
    'VGGc1024d4A': [1024, 'A', 1024, 1024, 1024],

    'VGGs': [8, 'M', 16, 'M', 32, 'M'],  # 's' for shallow.
    'VGGsw': [64, 'M', 128, 'M', 256, 'M'],  # 'w' for wide.
    'VGGsxw': [128, 'M', 256, 'M', 512, 'M'],  # 'x' for extra.

    'VGG8c': [64, 128, 'M', 256, 256, 'M', 512, 512, 'M'],

    # Models taken from the paper "Training Neural Networks with Local Error Signals"
    # https://github.com/anokland/local-loss/blob/master/train.py#L1276
    'VGG8b': [128, 256, 'M', 256, 512, 'M', 512, 'M', 512, 'M'],
    'VGG11b': [128, 128, 128, 256, 'M', 256, 512, 'M', 512, 512, 'M', 512, 'M'],

    # These are versions similar to the original VGG models but with less down-sampling,
    # reaching final spatial size of 4x4 instead of 1x1 in the original VGG architectures.
    # 'c' stands for CIFAR, i.e. models that are suited to CIFAR instead of ImageNet.
    'VGG11c': [64, 128, 'M', 256, 256, 'M', 512, 512, 512, 512, 'M'],
    'VGG13c': [64, 64, 128, 128, 'M', 256, 256, 'M', 512, 512, 512, 512, 'M'],
    'VGG16c': [64, 64, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512, 'M'],
    'VGG19c': [64, 64, 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512, 512, 512, 'M'],

    # Original VGG architectures (built for ImageNet images of size 224x224)
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def get_ssl_aux_net(channels: int,
                    image_size: Optional[int] = None,
                    target_image_size: Optional[int] = None) -> nn.Sequential:
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


def get_list_of_arguments_for_config(config: List[Union[int, str]], arg) -> list:
    """Given an argument `arg`, returns a list of arguments for a given config of a VGG model,
    where this argument is repeated for conv blocks (and None in the positions of non-conv blocks).
    """
    if isinstance(arg, list):
        non_conv_indices = [i for i, l in enumerate(config) if isinstance(l, str)]
        for non_conv_index in non_conv_indices:
            arg.insert(non_conv_index, None)
    else:
        arg = [(arg if isinstance(l, int) else None) for l in range(len(config))]
    
    return arg


def get_pool_layer(letter):
    """
    Returns:
        A pooling layer with kernel_size 2 and stride 2 - AvgPool if `letter` is 'A' and MaxPool if `letter` is 'M'.
    """
    if letter == 'M':
        pool_type = nn.MaxPool2d
    elif letter == 'A':
        pool_type = nn.AvgPool2d
    else:
        raise NotImplementedError(f'{letter=} is not supported, should be A or M.')

    return pool_type(kernel_size=2, stride=2)


def get_vgg_blocks(config: List[Union[int, str]],
                   in_channels: int = 3,
                   spatial_size: int = 32,
                   kernel_size: Union[int, List[int]] = 3,
                   padding: Union[int, List[int]] = 1,
                   use_batch_norm: Union[bool, List[bool]] = False,
                   bottleneck_dim: Union[int, List[int]] = 0,
                   pool_as_separate_blocks: bool = True,
                   shuffle_blocks_output: Union[bool, List[bool]] = False,
                   spatial_shuffle_only: Union[bool, List[bool]] = False,
                   fixed_permutation_per_block: Union[bool, List[bool]] = False) -> Tuple[List[nn.Module], int]:
    """Gets a list containing the blocks of the given VGG model config.

    Args:
        config: One of the lists in the dictionary `configs` above, describing the architecture of the network.
        in_channels: Number of input channels (3 for RGB images).
        spatial_size: The size of the input tensor for the network (32 for CIFAR10, 28 for MNIST, etc).
        kernel_size: The kernel size to use in each conv block. If it's a single variable, the same one is used.
        padding: The amount of padding to use in each conv block. If it's a single variable, the same one is used.
        use_batch_norm: Whether to use batch-norm in each conv block. If it's a single variable, the same one is used.
        bottleneck_dim: The dimension of the bottleneck layer to use in the end of each conv block
            (0 means no bottleneck is added). If it's a single variable, the same one is used.
        pool_as_separate_blocks: Whether to put the (avg/max) pool layers as separate blocks,
            or in the end of the previous conv block.
        shuffle_blocks_output: If it's true - shuffle the input for each block in the network.
            The input will be shuffled spatially only, meaning that the channels dimension will stay intact.
            For example, if the input is of shape 28x28x64 a random permutation from all (28*28)! possibilities
            is sampled and applied to the input tensor.
        spatial_shuffle_only: Shuffle the spatial locations only and the channels dimension will stay intact.
        fixed_permutation_per_block: A fixed permutation will be used every time this module is called.
    Returns:
        A tuple containing the list of nn.Modules, and an integers which is the number of output features
        (will be useful later when feeding to a linear layer).
    """
    blocks: List[nn.Module] = list()
    out_channels = in_channels
    
    kernel_size = get_list_of_arguments_for_config(config, kernel_size)
    padding = get_list_of_arguments_for_config(config, padding)
    use_batch_norm = get_list_of_arguments_for_config(config, use_batch_norm)
    bottleneck_dim = get_list_of_arguments_for_config(config, bottleneck_dim)
    shuffle_blocks_output = get_list_of_arguments_for_config(config, shuffle_blocks_output)
    spatial_shuffle_only = get_list_of_arguments_for_config(config, spatial_shuffle_only)
    fixed_permutation_per_block = get_list_of_arguments_for_config(config, fixed_permutation_per_block)

    for i in range(len(config)):
        if isinstance(config[i], str):
            spatial_size /= 2
            assert spatial_size == int(spatial_size), f'spatial_size is not an integer after dividing by 2'
            spatial_size = int(spatial_size)

            if pool_as_separate_blocks:
                blocks.append(nn.Sequential(get_pool_layer(config[i])))
            continue

        block_layers: List[nn.Module] = list()
        out_channels = config[i]

        block_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size[i], padding=padding[i]))
        if use_batch_norm[i]:
            block_layers.append(nn.BatchNorm2d(out_channels))
        block_layers.append(nn.ReLU())
        if (i + 1 < len(config)) and (isinstance(config[i + 1], str)) and (not pool_as_separate_blocks):
            block_layers.append(get_pool_layer(config[i + 1]))
        if bottleneck_dim[i] > 0:
            block_layers.append(nn.Conv2d(out_channels, bottleneck_dim[i], kernel_size=1))
            out_channels = bottleneck_dim[i]
        if shuffle_blocks_output[i]:
            block_layers.append(ShuffleTensor(spatial_size, out_channels,
                                              spatial_shuffle_only[i], fixed_permutation_per_block[i]))

        blocks.append(nn.Sequential(*block_layers))

        in_channels = out_channels
        spatial_size = spatial_size + 2 * padding[i] - kernel_size[i] + 1

    n_features = int(out_channels * (spatial_size ** 2))
    return blocks, n_features


def get_blocks(config: List[Union[int, str]],
               dropout_prob: float = 0,
               padding_mode: str = 'zeros',
               aux_mlp_n_hidden_layers: int = 1,
               aux_mlp_hidden_dim: int = 1024,
               final_mlp_n_hidden_layers: int = 1,
               final_mlp_hidden_dim: int = 1024,
               upsample: bool = False,
               pred_aux_type: str = 'mlp') -> Tuple[List[nn.Module],
                                                    List[Optional[nn.Module]],
                                                    List[Optional[nn.Module]],
                                                    int]:
    """
    Return a list of `blocks` which constitute the whole network,
    as well as lists of the auxiliary networks (both for prediction and for SSL).
    Each block is a sequence of several layers (Conv, BatchNorm, ReLU, MaxPool2d and Dropout).

    :param config: A list of integers / 'M' describing the architecture (see examples in the top of the file).
    :param aux_mlp_n_hidden_layers: Number of hidden layers in each prediction auxiliary network.
    :param aux_mlp_hidden_dim: The dimension of each hidden layer in each prediction auxiliary network.
    :param final_mlp_n_hidden_layers: Number of hidden layers in the final MLP sub-network (predicting the scores).
    :param final_mlp_hidden_dim: The dimension of each hidden layer in the final MLP sub-network.
    :param dropout_prob: When positive, add dropout after each non-linearity.
    :param padding_mode: Should be 'zeros' or 'circular' indicating the padding mode of each Conv layer.
    :param upsample: If true, upsample each SSL auxiliary network output to 32 x 32.
    :param pred_aux_type: 'mlp' for multi-layer-perceptron, 'cnn' for a single conv layer followed by mlp.
    :return: The blocks, the prediction auxiliary networks, the prediction auxiliary networks,
             and the dimension of the last layer (will be useful when feeding into a liner layer later).
    """
    blocks: List[nn.Module] = list()
    pred_auxiliary_nets: List[Optional[nn.Module]] = list()
    ssl_auxiliary_nets: List[Optional[nn.Module]] = list()

    in_channels = 3
    image_size = 32
    block_output_dimension = 0

    aux_mlp_kwargs = dict(output_dim=len(CLASSES),
                          n_hidden_layers=aux_mlp_n_hidden_layers,
                          hidden_dim=aux_mlp_hidden_dim)
    final_mlp_kwargs = dict(output_dim=len(CLASSES),
                            n_hidden_layers=final_mlp_n_hidden_layers,
                            hidden_dim=final_mlp_hidden_dim)
    for i in range(len(config)):
        if config[i] == 'M':
            blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            pred_auxiliary_nets.append(None)
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

            # In the last layer use MLP anyway (one before last because the actual last one is pool).
            if i == len(config) - 2:
                pred_aux_net = get_mlp(input_dim=block_output_dimension, **final_mlp_kwargs)
            elif pred_aux_type == 'mlp':
                pred_aux_net = get_mlp(input_dim=block_output_dimension, **aux_mlp_kwargs)
            else:  # pred_aux_type == 'cnn'
                pred_aux_net = get_cnn(conv_channels=[out_channels],
                                       linear_channels=[aux_mlp_hidden_dim] * aux_mlp_n_hidden_layers,
                                       spatial_size=image_size, in_channels=out_channels)
            pred_auxiliary_nets.append(pred_aux_net)

            upsampling_kwargs = dict(image_size=ssl_input_image_size, target_image_size=32) if upsample else dict()
            ssl_auxiliary_nets.append(get_ssl_aux_net(channels=out_channels, **upsampling_kwargs))

    return blocks, pred_auxiliary_nets, ssl_auxiliary_nets, block_output_dimension


class VGG(nn.Module):
    def __init__(self,
                 vgg_name: str,
                 final_mlp_n_hidden_layers: int = 1,
                 final_mlp_hidden_dim: int = 1024,
                 dropout_prob: float = 0,
                 padding_mode: str = 'zeros'):
        """
        Constructs a new VGG model, that is trained in a regular end-to-end fashion.

        :param vgg_name: Should be a key in the `configs` in the head of this file.
                         This describes the architecture of the desired model.
        :param final_mlp_n_hidden_layers: Number of hidden layers in the final MLP sub-network (predicting the scores).
        :param final_mlp_hidden_dim: The dimension of each hidden layer in the final MLP sub-network.
        :param dropout_prob: When positive, add dropout after each non-linearity.
        :param padding_mode: Should be 'zeros' or 'circular' indicating the padding mode of each Conv layer.
        """
        super(VGG, self).__init__()
        layers, _, _, features_output_dimension = get_blocks(configs[vgg_name], dropout_prob, padding_mode)
        self.features = nn.Sequential(*layers)
        self.mlp = get_mlp(input_dim=features_output_dimension, output_dim=len(CLASSES),
                           n_hidden_layers=final_mlp_n_hidden_layers, hidden_dimensions=final_mlp_hidden_dim)

    def forward(self, x: torch.Tensor):
        features = self.features(x)
        outputs = self.mlp(features)
        return outputs


def get_vgg_model_kernel_size(model, block_index: int):
    if not (0 <= block_index < len(model.features)):
        raise IndexError(f"block_index {block_index} is out-of-bounds (len={len(model.features)})")

    block = model.features[block_index]

    if not isinstance(block, nn.Sequential):
        raise ValueError(f"block_index {block_index} is not a sequential module (i.e. \'block\'), it's {type(block)}.")

    first_layer = block[0]

    if not any(isinstance(first_layer, cls) for cls in [nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d]):
        raise ValueError(f"first layer of the block is not a Conv/MaxPool/AvgPool layer, it's {type(first_layer)}")

    return first_layer.kernel_size


class VGGwDGL(nn.Module):
    def __init__(self,
                 vgg_name: str,
                 final_mlp_n_hidden_layers: int = 1,
                 final_mlp_hidden_dim: int = 1024,
                 dropout_prob: float = 0,
                 padding_mode: str = 'zeros',
                 pred_aux_type: str = 'cnn',
                 aux_mlp_n_hidden_layers: int = 1,
                 aux_mlp_hidden_dim: int = 1024,
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
                                                                       dropout_prob,
                                                                       padding_mode,
                                                                       aux_mlp_n_hidden_layers,
                                                                       aux_mlp_hidden_dim,
                                                                       final_mlp_n_hidden_layers,
                                                                       final_mlp_hidden_dim,
                                                                       upsample,
                                                                       pred_aux_type)
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

        # For the scores prediction aux-net, feed the representation after max-pooling (if it's indeed the next layer).
        next_is_pool = ((last_block_index+1 < len(self.blocks)) and
                        isinstance(self.blocks[last_block_index+1], nn.MaxPool2d))
        scores_aux_net_input = self.blocks[last_block_index+1](representation) if next_is_pool else representation

        scores_aux_net = self.auxiliary_nets[last_block_index]
        ssl_aux_net = self.ssl_auxiliary_nets[last_block_index] if (self.ssl_auxiliary_nets is not None) else None

        scores_outputs = scores_aux_net(scores_aux_net_input) if (scores_aux_net is not None) else None
        ssl_outputs = ssl_aux_net(representation) if (ssl_aux_net is not None) else None

        if self.ssl_auxiliary_nets is None:
            return representation, scores_outputs  # This mode trains only DGL, nothing to return as ssl_outputs.
        else:
            return representation, scores_outputs, ssl_outputs


class VGGwLG(nn.Module):
    def __init__(self,
                 vgg_name: str,
                 final_mlp_n_hidden_layers: int = 1,
                 final_mlp_hidden_dim: int = 1024,
                 dropout_prob: float = 0,
                 padding_mode: str = 'zeros',
                 pred_aux_type: str = 'cnn',
                 aux_mlp_n_hidden_layers: int = 1,
                 aux_mlp_hidden_dim: int = 1024):
        super(VGGwLG, self).__init__()
        blocks, auxiliary_networks, _, _ = get_blocks(configs[vgg_name],
                                                      dropout_prob,
                                                      padding_mode,
                                                      aux_mlp_n_hidden_layers,
                                                      aux_mlp_hidden_dim,
                                                      final_mlp_n_hidden_layers,
                                                      final_mlp_hidden_dim,
                                                      pred_aux_type=pred_aux_type)
        self.blocks = nn.ModuleList(blocks)
        self.auxiliary_nets = nn.ModuleList(auxiliary_networks)

    def forward(self, x: torch.Tensor):
        representation: torch.Tensor = x
        aux_nets_outputs: List[Optional[torch.Tensor]] = list()
        for i in range(len(self.blocks)):
            representation = self.blocks[i](representation)
            if self.auxiliary_nets[i] is None:
                aux_nets_outputs.append(None)
                continue

            # Feed the representation after max-pooling (if it's indeed the next layer).
            next_is_pool = (i+1 < len(self.blocks)) and isinstance(self.blocks[i+1], nn.MaxPool2d)
            scores_aux_net_input = self.blocks[i+1](representation) if next_is_pool else representation
            outputs = self.auxiliary_nets[i](scores_aux_net_input)
            aux_nets_outputs.append(outputs)

            representation = representation.detach()
        
        return aux_nets_outputs
