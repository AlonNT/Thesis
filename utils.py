import copy
import time

import torch
import torchvision
import itertools
import wandb

import numpy as np

from torchvision.transforms.functional import resize
from typing import List, Union, Optional, Dict
from loguru import logger

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
    # Begins with a flatten layer. It's useful when the input is 4D from a conv layer, and harmless otherwise..
    layers: List[torch.nn.Module] = [torch.nn.Flatten()]

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
                                 Each block will contain Conv - BatchNorm - MaxPool - ReLU.
                                 Defaults to 128, 128, 128.
    :param affine_layers_channels: A list of integers containing the channels of each linear layer.
                                 Defaults to 256, 10.
    :return: A sequential model which is the constructed CNN.
    """
    if conv_layers_channels is None:
        conv_layers_channels = [128, 128, 128]
    if affine_layers_channels is None:
        affine_layers_channels = [256, len(CLASSES)]

    conv_kernel_size = 5
    padding = 2
    pool_kernel_size = 2
    image_size = 32

    layers: List[torch.nn.Module] = list()

    in_channels = 3  # 3 channels corresponding to RGB channels in the original images.
    for conv_layer_channels in conv_layers_channels:
        out_channels = conv_layer_channels

        layers.append(torch.nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=conv_kernel_size,
                                      padding=padding))
        layers.append(torch.nn.BatchNorm2d(out_channels))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool2d(kernel_size=pool_kernel_size))

        in_channels = out_channels

    layers.append(torch.nn.Flatten())

    down_sample_factor = 2 ** len(conv_layers_channels)
    spatial_size = image_size // down_sample_factor
    in_features = conv_layers_channels[-1] * (spatial_size ** 2)
    for i, affine_layer_channels in enumerate(affine_layers_channels):
        layers.append(torch.nn.Linear(in_features=in_features, out_features=affine_layer_channels))
        if i < len(affine_layers_channels) - 1:  # Do not append ReLU in the last affine layer.
            layers.append(torch.nn.ReLU())

        in_features = affine_layer_channels

    return torch.nn.Sequential(*layers)


def one_hot(indices: torch.Tensor, device: torch.device):
    """
    Convert a tensor containing indices to tensor containing one-hot vectors (corresponding to the indices).

    :param indices: A tensor containing indices.
    :param device: The device to work on.
    :return: A variable containing the one-hot vectors.
    """
    result = torch.FloatTensor(indices.size() + (len(CLASSES),)).to(device)
    result.zero_()
    indices_rank = len(indices.size())
    result.scatter_(dim=indices_rank, index=indices.data.unsqueeze(dim=indices_rank), value=1)
    return torch.autograd.Variable(result)


def get_dataloaders(batch_size: int = 32,
                    normalize_to_unit_gaussian: bool = False,
                    normalize_to_plus_minus_one: bool = False,
                    random_crop: bool = False,
                    random_horizontal_flip: bool = False,
                    random_erasing: bool = False):
    """
    Get dataloaders for the CIFAR10 dataset.
    :param batch_size: The size of the mini-batches to initialize the dataloaders.
    :param normalize_to_unit_gaussian: If true, normalize the values to be in the range [-1,1] (instead of [0,1]).
    :param normalize_to_plus_minus_one: If true, normalize the values to be a unit gaussian.
    :param random_crop: If true, performs padding of 4 followed by random crop.
    :param random_horizontal_flip: If true, performs random horizontal flip.
    :param random_erasing: If true, performs erase a random rectangle in the image.
                           See https://arxiv.org/pdf/1708.04896.pdf.
    :return: A dictionary mapping "train"/"test" to its dataloader.
    """
    transforms = {'train': list(), 'test': list()}
    if random_horizontal_flip:
        transforms['train'].append(torchvision.transforms.RandomHorizontalFlip())
    if random_crop:
        transforms['train'].append(torchvision.transforms.RandomCrop(size=32, padding=4))
    for t in ['train', 'test']:
        transforms[t].append(torchvision.transforms.ToTensor())
    if normalize_to_plus_minus_one or normalize_to_unit_gaussian:
        assert not (normalize_to_plus_minus_one and normalize_to_unit_gaussian)
        # For the different normalization values see:
        # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/7
        if normalize_to_unit_gaussian:
            normalization_values = [(0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)]
        else:
            normalization_values = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
        for t in ['train', 'test']:
            transforms[t].append(torchvision.transforms.Normalize(*normalization_values))
    if random_erasing:
        transforms['train'].append(torchvision.transforms.RandomErasing())

    datasets = {t: torchvision.datasets.CIFAR10(root='./data',
                                                train=(t == 'train'),
                                                transform=torchvision.transforms.Compose(transforms[t]),
                                                download=False)
                for t in ['train', 'test']}

    dataloaders = {t: torch.utils.data.DataLoader(datasets[t],
                                                  batch_size=batch_size,
                                                  shuffle=(t == 'train'),
                                                  num_workers=5)
                   for t in ['train', 'test']}

    return dataloaders


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


def perform_train_step_local(model, inputs, labels, criterion, optim,
                             training_step,
                             ssl: bool = False, ssl_criterion=None,
                             pred_loss_weight: float = 1, ssl_loss_weight: float = 0.1,
                             first_trainable_block: int = 0,
                             shift_ssl_labels: bool = False,
                             images_log_interval: int = 1000):
    """
    Perform a train-step for a model trained with DGL.
    The difference between the regular train-step and this one is that the model forward pass
    is done iteratively for each block in the model, performing backward pass and optimizer step for each block
    (using its corresponding auxiliary network).

    :param model: The model.
    :param inputs: The inputs.
    :param labels: The labels.
    :param criterion: The criterion.
    :param optim: The optimizer (or plural optimizers).
    :param training_step:
    :param ssl:
    :param ssl_criterion:
    :param pred_loss_weight:
    :param ssl_loss_weight:
    :param first_trainable_block:
    :param shift_ssl_labels:
    :param images_log_interval:
    :return: The loss of this train-step, as well as the predictions.
    """
    n_plot_images = 4
    inputs_size = inputs.size(-1)
    inputs_representation = torch.clone(inputs)
    loss, scores, predictions_loss, ssl_loss = None, None, None, None
    indices_to_plot = np.random.choice(inputs.size(0), size=n_plot_images, replace=False)

    if first_trainable_block < 0:
        first_trainable_block = len(model.blocks) + first_trainable_block
    assert 0 <= first_trainable_block < len(model.blocks), f"Invalid first_trainable_block ({first_trainable_block})."

    shifts = tuple(np.linspace(start=5, stop=16, num=len(model.blocks), dtype=int))

    for i in range(len(model.blocks)):
        inputs_representation, outputs, ssl_outputs = model(inputs_representation,
                                                            first_block_index=i, last_block_index=i)
        if outputs is not None:
            scores = outputs

        if i < first_trainable_block:
            continue

        if optim[i] is not None:  # This means the current block has trainable parameters to optimize.
            optim[i].zero_grad()

            # If not training with SSL, punish according to the predicted classes' scores.
            # Note that in the last block we always want to do that, and ignore SSL.
            # Since the block in all of the models is simply MaxPool layer,
            # by 'last block' we mean one before the last.
            if (not ssl) or (i+2 == len(model.blocks)):
                loss = criterion(scores, labels)
            else:  # Training with SSL and it's not the last block.
                ssl_outputs_size = ssl_outputs.size(-1)
                ssl_labels = inputs if ssl_outputs_size == inputs_size else resize(inputs, size=[ssl_outputs_size] * 2)
                if shift_ssl_labels:
                    ssl_labels = torch.roll(ssl_labels, shifts=(shifts[i], shifts[i]), dims=(2, 3))
                ssl_loss = ssl_criterion(ssl_outputs, ssl_labels)

                if pred_loss_weight > 0:
                    predictions_loss = criterion(scores, labels)
                    loss = pred_loss_weight * predictions_loss + ssl_loss_weight * ssl_loss
                else:
                    loss = ssl_loss

                if training_step % images_log_interval == 0:
                    images = torch.cat([ssl_labels[indices_to_plot], ssl_outputs[indices_to_plot].detach()])
                    grid = torchvision.utils.make_grid(images, nrow=n_plot_images)
                    wandb_image = wandb.Image(grid.cpu().numpy().transpose((1, 2, 0)))
                    wandb.log({f'SSL-layer#{i}': [wandb_image]}, step=training_step)

            loss.backward()
            optim[i].step()

        inputs_representation = inputs_representation.detach()  # Prepare the input tensor for the next block.

    _, predictions = torch.max(scores, dim=1)
    return loss, predictions, predictions_loss, ssl_loss


def perform_train_step_cdni(model, inputs, labels, criterion, optim):
    """
    Perform a train-step for a model trained with cDNI.
    The difference between the regular train-step and this one is that the model forward pass
    needs to be fed with the labels (to be used as context for the synthetic gradients synthesizers),
    in addition to the inputs.

    :param model: The model.
    :param inputs: The inputs.
    :param labels: The labels.
    :param criterion: The criterion.
    :param optim: The optimizer (or plural optimizers).
    :return: The loss of this train-step, as well as the predictions.
    """
    optim.zero_grad()

    outputs = model(inputs, labels)
    _, predictions = torch.max(outputs, 1)
    loss = criterion(outputs, labels)

    loss.backward()
    optim.step()

    return loss, predictions


def perform_train_step_regular(model, inputs, labels, criterion, optim):
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
    :param optim: The optimizer (or plural optimizers).
    :return: The loss of this train-step, as well as the predictions.
    """
    optim.zero_grad()

    outputs = model(inputs)
    _, predictions = torch.max(outputs, 1)
    loss = criterion(outputs, labels)

    loss.backward()
    optim.step()

    return loss, predictions


def perform_train_step(model, inputs, labels, criterion, optim,
                       train_step, is_dgl, is_cdni, ssl, ssl_criterion,
                       pred_loss_weight, ssl_loss_weight, first_trainable_block, shift_ssl_labels):
    """
    Perform a single train-step, which is done differently when using regular training, DGL and cDNI.

    :param model: The model.
    :param inputs: The inputs.
    :param labels: The labels.
    :param criterion: The criterion.
    :param optim: The optimizer (or plural optimizers).
    :param train_step:
    :param is_dgl: Whether this model is trained using DGL (affects the optimizer and the train-step functionality).
    :param is_cdni: Whether this model is trained using DNI with context or not (affects the train-step functionality).
    :param ssl:
    :param ssl_criterion:
    :param pred_loss_weight:
    :param ssl_loss_weight:
    :param first_trainable_block:
    :param shift_ssl_labels:
    :return: The loss of this train-step, as well as the predictions.
    """
    mutual_args = (model, inputs, labels, criterion, optim)

    if is_dgl:
        return perform_train_step_local(*mutual_args, train_step, ssl, ssl_criterion, pred_loss_weight, ssl_loss_weight,
                                        first_trainable_block, shift_ssl_labels)
    elif is_cdni:
        return perform_train_step_cdni(*mutual_args)
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


class Accumulator:
    """
    Accumulate loss and correct predictions of an interval,
    to calculate later mean-loss & accuracy.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_sum: float = 0
        self.pred_loss_sum: float = 0
        self.ssl_loss_sum: float = 0
        self.corrects_sum: int = 0
        self.total_samples: int = 0
        self.begin_time: float = time.time()

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

    def get_dict(self, prefix='') -> Dict:
        return {f'{prefix}_accuracy': self.get_accuracy(),
                f'{prefix}_loss': self.get_mean_loss(),
                f'{prefix}_pred_loss': self.get_mean_pred_loss(),
                f'{prefix}_ssl_loss': self.get_mean_ssl_loss()}


def train_model(model, criterion, optimizer_params, dataloaders, device,
                num_epochs=25, log_interval=100, is_dgl=False, is_cdni=False,
                is_ssl=False, ssl_criterion=None, pred_loss_weight=1, ssl_loss_weight=0.1,
                first_trainable_block=0, shift_ssl_labels=False):
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
    :param pred_loss_weight:
    :param ssl_loss_weight:
    :param first_trainable_block:
    :param shift_ssl_labels:
    :return: the model with the lowest test error
    """
    best_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    optim: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]] = get_optim(model, optimizer_params, is_dgl)

    training_step = 0
    interval_accumulator = Accumulator()
    epoch_accumulator = Accumulator()

    for epoch in range(num_epochs):
        # model_state = copy.deepcopy(model.state_dict())  # For debugging purposes later - verify weights change.
        model.train()
        epoch_accumulator.reset()

        for inputs, labels in dataloaders['train']:
            training_step += 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            train_step_result = perform_train_step(model, inputs, labels, criterion, optim, training_step,
                                                   is_dgl, is_cdni,
                                                   is_ssl, ssl_criterion, pred_loss_weight, ssl_loss_weight,
                                                   first_trainable_block, shift_ssl_labels)

            accumulator_kwargs = dict()
            if is_dgl:
                loss, predictions, predictions_loss, ssl_loss = train_step_result
                accumulator_kwargs['mean_pred_loss'] = predictions_loss.item() if predictions_loss is not None else 0
                accumulator_kwargs['mean_ssl_loss'] = ssl_loss.item() if ssl_loss is not None else 0
            else:
                loss, predictions = train_step_result

            accumulator_kwargs['mean_loss'] = loss.item()
            accumulator_kwargs['num_corrects'] = torch.sum(torch.eq(predictions, labels.data)).item()
            accumulator_kwargs['n_samples'] = inputs.size(0)  # This equals the batch-size, except in the last minibatch

            epoch_accumulator.update(**accumulator_kwargs)
            interval_accumulator.update(**accumulator_kwargs)

            if training_step % log_interval == 0:
                wandb.log(data=interval_accumulator.get_dict(prefix='train'), step=training_step)
                interval_accumulator.reset()

        epoch_test_loss, epoch_test_accuracy = evaluate_model(model, criterion, dataloaders['test'], device)
        wandb.log(data={'test_accuracy': epoch_test_accuracy, 'test_loss': epoch_test_loss}, step=training_step)

        # if the current model reached the best results so far, deep copy the weights of the model.
        if epoch_test_accuracy > best_accuracy:
            best_accuracy = epoch_test_accuracy
            best_weights = copy.deepcopy(model.state_dict())

        epoch_time_elapsed = epoch_accumulator.get_time()
        logger.info(f'Epoch {epoch + 1:0>2d}/{num_epochs:0>2d} | '
                    f'{int(epoch_time_elapsed // 60):d}m {int(epoch_time_elapsed) % 60:0>2d}s | '
                    f'Train '
                    f'loss={epoch_accumulator.get_mean_loss():.4f} '
                    f'accuracy={epoch_accumulator.get_accuracy():.2f}% | '
                    f'Test '
                    f'loss={epoch_test_loss:.4f} '
                    f'accuracy={epoch_test_accuracy:.2f}%')

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
