import copy
import time

import torch
import torch.nn as nn
import torchvision
import itertools
import wandb

import numpy as np

from torchvision.transforms.functional import resize
from typing import List, Union, Optional, Dict
from loguru import logger
from datetime import timedelta

from consts import CLASSES


class Accumulator:
    """
    Accumulate loss and correct predictions of an interval,
    to calculate later mean-loss & accuracy.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_sum: float = 0
        self.corrects_sum: int = 0
        self.total_samples: int = 0
        self.begin_time: float = time.time()

        # These are for training with SSL & DGL combined, to examine the two different objectives separately.
        self.pred_loss_sum: float = 0
        self.ssl_loss_sum: float = 0

    def update(self, mean_loss: float, num_corrects: int, n_samples: int,
               mean_pred_loss: float = 0, mean_ssl_loss: float = 0):
        self.loss_sum += mean_loss * n_samples
        self.pred_loss_sum += mean_pred_loss * n_samples
        self.ssl_loss_sum += mean_ssl_loss * n_samples
        self.corrects_sum += num_corrects
        self.total_samples += n_samples

    def get_mean_loss(self) -> float:
        return self.loss_sum / self.total_samples

    def get_mean_pred_loss(self) -> float:
        return self.pred_loss_sum / self.total_samples

    def get_mean_ssl_loss(self) -> float:
        return self.ssl_loss_sum / self.total_samples

    def get_accuracy(self) -> float:
        return 100 * (self.corrects_sum / self.total_samples)

    def get_time(self) -> float:
        return time.time() - self.begin_time

    def get_dict(self, prefix='') -> Dict[str, float]:
        d = {f'{prefix}_accuracy': self.get_accuracy(),
             f'{prefix}_loss': self.get_mean_loss()}

        if self.get_mean_pred_loss() > 0:
            d[f'{prefix}_pred_loss'] = self.get_mean_pred_loss()
        if self.get_mean_ssl_loss() > 0:
            d[f'{prefix}_ssl_loss'] = self.get_mean_ssl_loss()

        return d


def cross_entropy_gradient(logits, labels):
    """
    Calculate the gradient of the cross-entropy loss with respect to the input logits.
    Note the cross-entropy loss in PyTorch basically calculates log-softmax followed by negative log-likelihood loss.
    Therefore, the gradient is the softmax output of the logits, where in the labels indices a 1 is subtracted.

    Inspiration from http://machinelearningmechanic.com/deep_learning/2019/09/04/cross-entropy-loss-derivative.html

    :param logits: The raw scores which are the input to the cross-entropy-loss.
    :param labels: The labels (for each i the index of the true class of this training-sample).
    :return: The gradient of the cross-entropy loss.
    """
    # This is the probabilities vector obtained using the softmax function on the raw scores.
    p = torch.nn.functional.softmax(logits, dim=1)

    # Subtract 1 from the labels indices, which gives the final gradient of the cross-entropy loss.
    p.scatter_add_(dim=1, index=labels.unsqueeze(dim=-1), src=torch.full_like(p, fill_value=-1))

    # TODO For some reason this differ from the gradient computed by torch.nn.CrossEntropyLoss (checked with hooks).

    return p


# PyTorch 1.1.0 (compatible with CUDA 9.0) does not have a Flatten layer, so it's copied here.
class Flatten(nn.Module):
    r"""
    Flattens a contiguous range of dims into a tensor. For use with :class:`~nn.Sequential`.

    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, \prod *dims)` (for the default case).

    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).
    """
    __constants__ = ['start_dim', 'end_dim']
    start_dim: int
    end_dim: int

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor.flatten(self.start_dim, self.end_dim)

    def extra_repr(self) -> str:
        return 'start_dim={}, end_dim={}'.format(
            self.start_dim, self.end_dim
        )


def get_mlp(input_dim: int, output_dim: int, n_hidden_layers: int = 0, hidden_dim: int = 0) -> torch.nn.Sequential:
    """
    This function builds a MLP (i.e. Multi-Layer-Perceptron) and return it as a PyTorch's sequential model.

    :param input_dim: The dimension of the input tensor.
    :param output_dim: The dimension of the output tensor.
    :param n_hidden_layers: Number of hidden layers.
    :param hidden_dim: The dimension of each hidden layer.
    :return: A sequential model which is the constructed MLP.
    """
    # Begins with a flatten layer. It's useful when the input is 4D from a conv layer, and harmless otherwise..
    layers: List[torch.nn.Module] = [Flatten()]

    for i in range(n_hidden_layers):
        layers.append(torch.nn.Linear(in_features=input_dim if i == 0 else hidden_dim,
                                      out_features=hidden_dim))
        layers.append(torch.nn.ReLU())

    layers.append(torch.nn.Linear(in_features=input_dim if n_hidden_layers == 0 else hidden_dim,
                                  out_features=output_dim))

    return torch.nn.Sequential(*layers)


def get_cnn(conv_layers_channels: List[int], affine_hidden_layers_channels: List[int],
            image_size: int = 32, in_channels: int = 3) -> torch.nn.Sequential:
    """
    This function builds a CNN and return it as a PyTorch's sequential model.

    :param conv_layers_channels: A list of integers containing the channels of each convolution block.
                                 Each block will contain Conv - BatchNorm - MaxPool - ReLU.
    :param affine_hidden_layers_channels: A list of integers containing the channels of each linear hidden layer.
    :param image_size: Will be used to infer input dimension for the first affine layer.
    :param in_channels: Number of channels in the input tensor.
    :return: A sequential model which is the constructed CNN.
    """
    layers: List[torch.nn.Module] = list()

    for n_channels in conv_layers_channels:
        out_channels = n_channels

        layers.append(torch.nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=3,
                                      padding=1))
        layers.append(torch.nn.BatchNorm2d(out_channels))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))

        in_channels = out_channels

    layers.append(Flatten())

    down_sample_factor = 2 ** len(conv_layers_channels)
    spatial_size = image_size // down_sample_factor
    in_features = conv_layers_channels[-1] * (spatial_size ** 2)
    for i, n_channels in enumerate(affine_hidden_layers_channels):
        out_features = n_channels
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.ReLU())
        in_features = out_features

    layers.append(torch.nn.Linear(in_features, len(CLASSES)))

    return torch.nn.Sequential(*layers)


def get_dataloaders(batch_size: int = 64,
                    normalize_to_unit_gaussian: bool = False,
                    normalize_to_plus_minus_one: bool = False,
                    random_crop: bool = False,
                    random_horizontal_flip: bool = False,
                    random_erasing: bool = False,
                    random_resized_crop: bool = False):
    """
    Get dataloaders for the CIFAR10 dataset, including data augmentations as requested by the arguments.

    :param batch_size: The size of the mini-batches to initialize the dataloaders.
    :param normalize_to_unit_gaussian: If true, normalize the values to be a unit gaussian.
    :param normalize_to_plus_minus_one: If true, normalize the values to be in the range [-1,1] (instead of [0,1]).
    :param random_crop: If true, performs padding of 4 followed by random crop.
    :param random_horizontal_flip: If true, performs random horizontal flip.
    :param random_erasing: If true, performs erase a random rectangle in the image.
                           See https://arxiv.org/pdf/1708.04896.pdf.
    :param random_resized_crop: If true, performs random resized crop.
    :return: A dictionary mapping "train"/"test" to its dataloader.
    """
    transforms = {'train': list(), 'test': list()}

    if random_horizontal_flip:
        transforms['train'].append(torchvision.transforms.RandomHorizontalFlip())
    if random_crop:
        transforms['train'].append(torchvision.transforms.RandomCrop(size=32, padding=4))
    if random_resized_crop:
        transforms['train'].append(torchvision.transforms.RandomResizedCrop(size=32, scale=(0.75, 1.), ratio=(1., 1.)))
    for t in ['train', 'test']:
        transforms[t].append(torchvision.transforms.ToTensor())
    if random_erasing:
        transforms['train'].append(torchvision.transforms.RandomErasing())
    if normalize_to_plus_minus_one or normalize_to_unit_gaussian:
        # For the different normalization values see:
        # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/7
        if normalize_to_unit_gaussian:
            normalization_values = [(0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)]
        else:
            normalization_values = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
        for t in ['train', 'test']:
            transforms[t].append(torchvision.transforms.Normalize(*normalization_values))

    datasets = {t: torchvision.datasets.CIFAR10(root='./data',
                                                train=(t == 'train'),
                                                transform=torchvision.transforms.Compose(transforms[t]),
                                                download=False)
                for t in ['train', 'test']}

    dataloaders = {t: torch.utils.data.DataLoader(datasets[t],
                                                  batch_size=batch_size,
                                                  shuffle=(t == 'train'),
                                                  num_workers=6)
                   for t in ['train', 'test']}

    return dataloaders


def evaluate_local_model(model, criterion, dataloader, device, training_step=None, log_to_wandb: bool = True):
    """
    Evaluate the given model on the test set.
    In addition to returning the final test loss & accuracy,
    this function evaluate each one of the model local modules (by logging to wandb).

    :param model: The model
    :param criterion: The criterion.
    :param dataloader: The test set data-loader.
    :param device: The device to use.
    :param training_step: The training-step (integer), important to wandb logging.
    :param log_to_wandb: Whether to log to wandb or not.
    :return: The test set loss and accuracy.
    """
    model.eval()
    n_modules = len(model.blocks)

    modules_accumulators = [Accumulator() if (aux_net is not None) else None for aux_net in model.auxiliary_nets]

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            inputs_representation = inputs
            for i in range(n_modules):
                result = model(inputs_representation, first_block_index=i, last_block_index=i)
                inputs_representation, outputs = result[0], result[1]
                if outputs is not None:
                    loss = criterion(outputs, labels)

                    modules_accumulators[i].update(
                        mean_loss=loss.item(),
                        num_corrects=torch.sum(torch.eq(torch.max(outputs, dim=1)[1], labels.data)).item(),
                        n_samples=inputs.size(0)
                    )

    if log_to_wandb:
        assert training_step is not None
        for i, modules_accumulator in enumerate(modules_accumulators):
            if modules_accumulator is not None:
                wandb.log(data=modules_accumulator.get_dict(prefix=f'module#{i}_test'), step=training_step)

    final_accumulator = modules_accumulators[-2]  # Last one is None because last block is MaxPool with no aux-net.
    return final_accumulator.get_mean_loss(), final_accumulator.get_accuracy()


def evaluate_model(model, criterion, dataloader, device):
    """
    Evaluate the given model on the test set.

    :param model: The model
    :param criterion: The criterion.
    :param dataloader: The test set data-loader.
    :param device: The device to use.
    :return: The test set loss and accuracy.
    """
    model.eval()

    loss_sum = 0.0
    corrects_sum = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            result = model(inputs)

            # In DGL the model forward function also return the inputs representation
            # (in addition to the classes' scores which are the prediction of the relevant auxiliary network)
            outputs = result[1] if isinstance(result, tuple) else result
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        loss_sum += loss.item() * inputs.size(0)
        corrects_sum += torch.sum(torch.eq(predictions, labels.data)).item()

    loss = loss_sum / len(dataloader.dataset)
    accuracy = 100 * (corrects_sum / len(dataloader.dataset))

    return loss, accuracy


def perform_train_step_dgl(model, inputs, labels, criterion, optimizers, training_step,
                           modules_accumulators: List[Optional[Accumulator]],
                           log_interval: int = 100):
    """
    Perform a train-step for a model trained with DGL.
    The difference between the regular train-step and this one is that the model forward pass
    is done iteratively for each block in the model, performing backward pass and optimizer step for each block
    (using its corresponding auxiliary network).

    :param model: The model.
    :param inputs: The inputs.
    :param labels: The labels.
    :param criterion: The criterion.
    :param optimizers: The optimizers (one for each local module in the whole model).
    :param training_step: The training-step (integer), important to wandb logging.
    :param modules_accumulators: Accumulators for each local module.
    :param log_interval: How many training/testing steps between each logging (to wandb).
    :return: The loss of this train-step, as well as the predictions.
    """
    inputs_representation = torch.clone(inputs)
    loss, predictions = None, None

    for i in range(len(model.blocks)):
        inputs_representation, outputs = model(inputs_representation, first_block_index=i, last_block_index=i)
        if outputs is not None:
            assert optimizers[i] is not None, "If the module has outputs it means it has an auxiliary-network " \
                                              "attached so it should has trainable parameters to optimize."
            predictions = torch.max(outputs, dim=1)[1]  # Save to return the predictions (argmax) later.
            optimizers[i].zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizers[i].step()

            modules_accumulators[i].update(
                mean_loss=loss.item(),
                num_corrects=torch.sum(torch.eq(predictions, labels.data)).item(),
                n_samples=inputs.size(0)
            )

        inputs_representation = inputs_representation.detach()  # Prepare the input tensor for the next block.

    if training_step % log_interval == 0:
        for i, modules_accumulator in enumerate(modules_accumulators):
            if modules_accumulator is not None:
                wandb.log(data=modules_accumulator.get_dict(prefix=f'module#{i}_train'), step=training_step)
                modules_accumulator.reset()

    return loss.item(), predictions


def perform_train_step_ssl(model, inputs, labels, scores_criterion, optimizers, training_step,
                           ssl_criterion=None,
                           pred_loss_weight: float = 1, ssl_loss_weight: float = 0.1,
                           first_trainable_block: int = 0,
                           shift_ssl_labels: bool = False,
                           images_log_interval: int = 1000):
    """
    Perform a train-step for a model trained with local self-supervised loss, possibly in combination with DGL.

    :param model: The model.
    :param inputs: The inputs.
    :param labels: The labels.
    :param scores_criterion: The criterion.
    :param optimizers: The optimizers.
    :param training_step: The training-step (integer), important to wandb logging.
    :param ssl_criterion: The loss for the SSL outputs (regression, e.g. L1 or L2).
    :param pred_loss_weight: When combining with DGL, the weight for the scores' prediction loss.
    :param ssl_loss_weight: When combining with DGL, the weight for the SSL prediction loss.
    :param first_trainable_block: Can be used to freeze a few layers in their initial weights.
    :param shift_ssl_labels: Whether to shift the SSL labels or keep them the original images.
    :param images_log_interval: Frequency of logging images to wandb
                                (less frequent then regular logging because it's quite heavy).
    :return: The loss of this train-step, as well as the predictions.
    """
    inputs_size = inputs.size(-1)
    inputs_representation = torch.clone(inputs)
    loss, predictions, scores_loss, ssl_loss = None, None, None, None
    n_plot_images = 4
    indices_to_plot = np.random.choice(inputs.size(0), size=n_plot_images, replace=False)
    shifts = tuple(np.linspace(start=5, stop=16, num=len(model.blocks), dtype=int))

    if first_trainable_block < 0:
        first_trainable_block = len(model.blocks) + first_trainable_block
    assert 0 <= first_trainable_block < len(model.blocks), f"Invalid first_trainable_block ({first_trainable_block})."

    for i in range(len(model.blocks)):
        inputs_representation, scores, ssl_outputs = model(inputs_representation,
                                                           first_block_index=i, last_block_index=i)
        if i < first_trainable_block:
            continue

        if scores is not None:
            assert ssl_outputs is not None
            assert optimizers[i] is not None, "If the module has outputs it means it has an auxiliary-network " \
                                              "attached so it should has trainable parameters to optimize."
            predictions = torch.max(scores, dim=1)[1]
            optimizers[i].zero_grad()

            # Note that in the last block we always want to punish according to the predicted classes' scores,
            # nd ignore SSL. Since the block in all of the models is simply MaxPool layer,
            # by 'last block' we mean one before the last.
            if i == len(model.blocks) - 2:
                loss = scores_criterion(scores, labels)
            else:
                ssl_outputs_size = ssl_outputs.size(-1)
                ssl_labels = inputs if ssl_outputs_size == inputs_size else resize(inputs, size=[ssl_outputs_size] * 2)
                if shift_ssl_labels:
                    ssl_labels = torch.roll(ssl_labels, shifts=(shifts[i], shifts[i]), dims=(2, 3))

                ssl_loss = ssl_criterion(ssl_outputs, ssl_labels)

                if pred_loss_weight > 0:
                    scores_loss = scores_criterion(scores, labels)
                    loss = pred_loss_weight * scores_loss + ssl_loss_weight * ssl_loss
                else:
                    loss = ssl_loss

                if training_step % images_log_interval == 0:
                    images = torch.cat([ssl_labels[indices_to_plot], ssl_outputs[indices_to_plot].detach()])
                    grid = torchvision.utils.make_grid(images, nrow=n_plot_images)
                    wandb_image = wandb.Image(grid.cpu().numpy().transpose((1, 2, 0)))
                    wandb.log({f'SSL-layer#{i}': [wandb_image]}, step=training_step)

            loss.backward()
            optimizers[i].step()
        else:
            assert ssl_outputs is None

        inputs_representation = inputs_representation.detach()  # Prepare the input tensor for the next block.

    return loss.item(), predictions, scores_loss, ssl_loss


def perform_train_step_direct_global(model, inputs, labels, criterion, optimizers,
                                     training_step, modules_accumulators, last_gradient_weight: float = 0.5,
                                     log_interval: int = 100):
    """
    Perform a train-step for a model trained with local self-supervised loss, possibly in combination with DGL.

    :param model: The model.
    :param inputs: The inputs.
    :param labels: The labels.
    :param criterion: The criterion.
    :param optimizers: The optimizers.
    :param training_step: The training-step (integer), important to wandb logging.
    :param modules_accumulators: Accumulators for each local module.
    :param last_gradient_weight: Weight of the last gradient in each intermediate gradient calculator.
    :param log_interval: How many training/testing steps between each logging (to wandb).
    :return: The loss of this train-step, as well as the predictions.
    """
    inputs_representation = torch.clone(inputs)
    loss_value, predictions = None, None

    modules_outputs: List[torch.Tensor] = list()

    # Perform the forward-pass.
    for i in range(len(model.blocks)):
        inputs_representation, outputs = model(inputs_representation, first_block_index=i, last_block_index=i)
        modules_outputs.append(outputs)

        if i == len(model.blocks) - 2:
            predictions = torch.max(outputs, dim=1)[1]

        inputs_representation = inputs_representation.detach()  # Prepare the input tensor for the next block.

    # Can't change variables within inner functions, but can change inner state of mutable variables.
    # https://stackoverflow.com/questions/11987358/
    last_layer_grad = dict(value=None)  # This will hold the gradient of the last layer.

    def last_module_hook(grad):
        assert last_layer_grad['value'] is None, "\'last_layer_grad\' should not have been set yet."
        last_layer_grad['value'] = grad

    def intermediate_module_hook(grad):
        assert last_layer_grad['value'] is not None, "\'last_layer_grad\' should have been set."
        return (1 - last_gradient_weight) * grad + last_gradient_weight * last_layer_grad['value']

    # Perform the backward-pass, using the gradients of the last layer w.r.t the outputs tensor.
    for i in range(len(model.blocks) - 1, -1, -1):
        module_outputs = modules_outputs[i]
        module_optimizer = optimizers[i]

        if module_outputs is not None:
            assert module_optimizer is not None
            module_optimizer.zero_grad()
            loss = criterion(module_outputs, labels)

            modules_accumulators[i].update(
                mean_loss=loss.item(),
                num_corrects=torch.sum(torch.eq(torch.max(module_outputs, dim=1)[1], labels.data)).item(),
                n_samples=inputs.size(0)
            )

            if last_layer_grad['value'] is None:
                assert i == len(model.blocks) - 2, "This should happen in the last block (that is not MaxPool)."
                module_outputs.register_hook(last_module_hook)
                loss_value = loss.item()  # This loss will be returned eventually, since it's the final model's loss.
            else:
                module_outputs.register_hook(intermediate_module_hook)

            loss.backward()
            module_optimizer.step()

    if training_step % log_interval == 0:
        for i, modules_accumulator in enumerate(modules_accumulators):
            if modules_accumulator is not None:
                wandb.log(data=modules_accumulator.get_dict(prefix=f'module#{i}_train'), step=training_step)
                modules_accumulator.reset()

    return loss_value, predictions


def perform_train_step_regular(model, inputs, labels, criterion, optimizer):
    """
    Perform a regular train-step:
    (1) Feed the inputs (i.e. minibatch images) to the model.
    (2) Get the predictions (i.e. classes' scores).
    (3) Calculate the loss with respect to the labels.
    (4) Perform backward pass and a single optimizer step.

    :param model: The model.
    :param inputs: The inputs.
    :param labels: The labels.
    :param criterion: The criterion.
    :param optimizer: The optimizer.
    :return: The loss of this train-step, as well as the predictions.
    """
    optimizer.zero_grad()

    outputs = model(inputs)
    _, predictions = torch.max(outputs, 1)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()

    return loss.item(), predictions


def perform_train_step(model, inputs, labels, criterion, optim,
                       training_step, is_dgl, ssl, ssl_criterion,
                       pred_loss_weight, ssl_loss_weight, first_trainable_block, shift_ssl_labels,
                       is_direct_global, modules_accumulators, last_gradient_weight):
    """
    Perform a single train-step, which is done differently when using regular training, DGL and cDNI.

    :param model: The model.
    :param inputs: The inputs.
    :param labels: The labels.
    :param criterion: The criterion.
    :param optim: The optimizer (or plural optimizers).
    :param training_step: The training-step (integer), important to wandb logging.
    :param is_dgl: Whether this model is trained using DGL (affects the optimizer and the train-step functionality).
    :param ssl: Whether this model is trained using SSL.
    :param ssl_criterion: The criterion for the SSL predictions.
    :param pred_loss_weight: When combining with DGL, the weight for the scores' prediction loss.
    :param ssl_loss_weight: When combining with DGL, the weight for the SSL prediction loss.
    :param first_trainable_block: Can be used to freeze a few layers in their initial weights.
    :param shift_ssl_labels: Whether to shift the SSL labels or keep them the original images.
    :param is_direct_global: Whether to use direct global gradient or not.
    :param modules_accumulators: Accumulators for each local module.
    :param last_gradient_weight: Weight of the last gradient in each intermediate gradient calculator.
    :return: The loss of this train-step, as well as the predictions.
    """
    mutual_args = (model, inputs, labels, criterion, optim)

    if is_dgl:
        if is_direct_global:
            return perform_train_step_direct_global(*mutual_args, training_step, modules_accumulators,
                                                    last_gradient_weight)
        elif ssl:
            return perform_train_step_ssl(*mutual_args, training_step, ssl_criterion, pred_loss_weight, ssl_loss_weight,
                                          first_trainable_block, shift_ssl_labels)
        else:
            return perform_train_step_dgl(*mutual_args, training_step, modules_accumulators)
    else:
        return perform_train_step_regular(*mutual_args)


def get_optim(model, optimizer_params, is_dgl):
    """
    Return the optimizer (or plural optimizers) to train the given model.
    If the model is trained with DGL there are several optimizers,
    one for each block in the model (which is optimizing the block
    parameters as well as the corresponding auxiliary network parameters).

    :param model: The model.
    :param optimizer_params: A dictionary describing the optimizer parameters: learning-rate, momentum and weight-decay.
    :param is_dgl: Whether this model is trained with DGL or not.
    :return: An optimizer, or a list of optimizers (if the model is trained with DGL).
    """
    optimizer_type: str = optimizer_params.pop('optimizer_type')
    if optimizer_type == 'Adam':
        optimizer_constuctor = torch.optim.Adam
        optimizer_params.pop('momentum')  # momentum is relevant only for SGD, not for Adam.
    elif optimizer_type == 'SGD':
        optimizer_constuctor = torch.optim.SGD
    else:
        raise ValueError(f'optimizer_type {optimizer_type} should be \'Adam\' or \'SGD\'.')

    if is_dgl:
        optimizers = list()

        for i in range(len(model.blocks)):
            if len(list(model.blocks[i].parameters())) == 0:
                optimizers.append(None)
            else:
                parameters_to_train = itertools.chain(
                    model.blocks[i].parameters(),
                    model.auxiliary_nets[i].parameters(),
                    model.ssl_auxiliary_nets[i].parameters() if model.ssl_auxiliary_nets is not None else list()
                )

                optimizer = optimizer_constuctor(parameters_to_train, **optimizer_params)
                optimizers.append(optimizer)

        return optimizers
    else:
        return optimizer_constuctor(model.parameters(), **optimizer_params)


def train_model(model, criterion, optimizer_params, dataloaders, device,
                num_epochs=25, log_interval=100, is_dgl=False, 
                is_ssl=False, ssl_criterion=None, pred_loss_weight=1, ssl_loss_weight=0.1,
                first_trainable_block=0, shift_ssl_labels=False,
                is_direct_global=False, last_gradient_weight: float = 0.5):
    """
    A general function to train a model and return the best model found.

    :param model: the model to train
    :param criterion: which loss to train on
    :param optimizer_params: the optimizer to train with
    :param dataloaders: the dataloaders to feed the model
    :param device: which device to train on
    :param num_epochs: how many epochs to train
    :param log_interval: How many training/testing steps between each logging (to wandb).
    :param is_dgl: Whether this model is trained using DGL (affects the optimizer and the train-step functionality).
    :param is_cdni: Whether this model is trained using DNI with context or not (affects the train-step functionality).
    :param is_ssl: Whether this model is trained using self-supervised local loss (predicting the shifted image).
    :param ssl_criterion:
    :param pred_loss_weight: When combining with DGL, the weight for the scores' prediction loss.
    :param ssl_loss_weight: When combining with DGL, the weight for the SSL prediction loss.
    :param first_trainable_block: Can be used to freeze a few layers in their initial weights.
    :param shift_ssl_labels: Whether to shift the SSL labels or keep them the original images.
    :param is_direct_global: Whether to use direct global gradient or not.
    :param last_gradient_weight: Weight of the last gradient in each intermediate gradient calculator.
    :return: the model with the lowest test error
    """
    best_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    optim: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]] = get_optim(model, optimizer_params, is_dgl)

    total_time = 0
    training_step = 0
    interval_accumulator = Accumulator()
    epoch_accumulator = Accumulator()

    if is_dgl:
        modules_accumulators = [Accumulator() if (aux_net is not None) else None for aux_net in model.auxiliary_nets]
    else:
        modules_accumulators = None

    for epoch in range(num_epochs):
        # model_state = copy.deepcopy(model.state_dict())  # For debugging purposes later - verify weights change.
        model.train()
        epoch_accumulator.reset()

        for inputs, labels in dataloaders['train']:
            training_step += 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            train_step_result = perform_train_step(model, inputs, labels, criterion, optim, training_step,
                                                   is_dgl, is_ssl, ssl_criterion, pred_loss_weight, ssl_loss_weight,
                                                   first_trainable_block, shift_ssl_labels, is_direct_global,
                                                   modules_accumulators, last_gradient_weight)

            accumulator_kwargs = dict()
            if is_ssl:
                loss, predictions, predictions_loss, ssl_loss = train_step_result
                accumulator_kwargs['mean_pred_loss'] = predictions_loss.item() if predictions_loss is not None else 0
                accumulator_kwargs['mean_ssl_loss'] = ssl_loss.item() if ssl_loss is not None else 0
            else:
                loss, predictions = train_step_result

            accumulator_kwargs['mean_loss'] = loss
            accumulator_kwargs['num_corrects'] = torch.sum(torch.eq(predictions, labels.data)).item()
            accumulator_kwargs['n_samples'] = inputs.size(0)  # This equals the batch-size, except in the last minibatch

            epoch_accumulator.update(**accumulator_kwargs)
            interval_accumulator.update(**accumulator_kwargs)

            if training_step % log_interval == 0:
                wandb.log(data=interval_accumulator.get_dict(prefix='train'), step=training_step)
                interval_accumulator.reset()

        if is_dgl:
            epoch_test_loss, epoch_test_accuracy = evaluate_local_model(model, criterion, dataloaders['test'], device,
                                                                        training_step)
        else:
            epoch_test_loss, epoch_test_accuracy = evaluate_model(model, criterion, dataloaders['test'], device)

        wandb.log(data={'test_accuracy': epoch_test_accuracy, 'test_loss': epoch_test_loss}, step=training_step)

        # if the current model reached the best results so far, deep copy the weights of the model.
        if epoch_test_accuracy > best_accuracy:
            best_accuracy = epoch_test_accuracy
            best_weights = copy.deepcopy(model.state_dict())

        epoch_time_elapsed = epoch_accumulator.get_time()
        total_time += epoch_time_elapsed
        epochs_left = num_epochs - (epoch + 1)
        avg_epoch_time = total_time / (epoch + 1)
        time_left = avg_epoch_time * epochs_left
        logger.info(f'Epoch {epoch + 1:0>3d}/{num_epochs:0>3d} '
                    f'({str(timedelta(seconds=epoch_time_elapsed)).split(".")[0]}) | '
                    f'ETA {str(timedelta(seconds=time_left)).split(".")[0]} | '
                    f'Train '
                    f'loss={epoch_accumulator.get_mean_loss():.4f} '
                    f'acc={epoch_accumulator.get_accuracy():.2f}% | '
                    f'Test '
                    f'loss={epoch_test_loss:.4f} '
                    f'acc={epoch_test_accuracy:.2f}%')

        # # For debugging purposes - verify that the weights of the model changed.
        # new_model_state = copy.deepcopy(model.state_dict())
        # for weight_name in new_model_state.keys():
        #     old_weight = model_state[weight_name]
        #     new_weight = new_model_state[weight_name]
        #     if torch.allclose(old_weight, new_weight):
        #         logger.warning(f'Weight \'{weight_name}\' of shape {list(new_weight.size())} did not change.')
        #     else:
        #         logger.debug(f'Weight \'{weight_name}\' of shape {list(new_weight.size())} changed.')
        # model_state = copy.deepcopy(new_model_state)

    logger.info(f'Best test accuracy: {best_accuracy:.2f}%')

    model.load_state_dict(best_weights)  # load best model weights
    return model
