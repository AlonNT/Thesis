import copy
import time

import torch
import torchvision
import itertools
import wandb

from typing import List, Union
from loguru import logger

from consts import CLASSES


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


def get_data(batch_size: int = 32):
    """
    Get the CIFAR10 data.
    :param batch_size: The size of the mini-batches to initialize the dataloaders.
    :return: image_datasets: Dictionary mapping "train"/"test" to its dataset
             dataloaders:    Dictionary mapping "train"/"test" to its dataloader
             dataset_sizes:  Dictionary mapping "train"/"test" to its dataset size
             classes:        tuple of 10 classes names in the correct order
    """
    # For the use of 0.5 for the normalization, see:
    # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/7
    image_datasets = {x: torchvision.datasets.CIFAR10(root='./data',
                                                      train=(x == 'train'),
                                                      transform=torchvision.transforms.Compose(
                                                          [torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                            (0.5, 0.5, 0.5))]
                                                      ),
                                                      download=False)  # Change to download=True in the first time.
                      for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=0)
                   for x in ['train', 'test']}

    return image_datasets, dataloaders, dataset_sizes


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


def perform_train_step_dgl(model, inputs, labels, criterion, optim):
    """
    Perform a train-step for a model trained with cDNI.
    The difference between the regular train-step and this one is that the model forward pass
    is done iteratively for each block in the model, performing backward pass and optimizer step for each block
    (using its corresponding auxiliary network).

    :param model: The model.
    :param inputs: The inputs.
    :param labels: The labels.
    :param criterion: The criterion.
    :param optim: The optimizer (or plural optimizers).
    :return: The loss of this train-step, as well as the predictions.
    """
    inputs_representation = inputs
    for i in range(len(model.blocks)):
        optim[i].zero_grad()

        inputs_representation, outputs = model(inputs_representation,
                                               first_block_index=i, last_block_index=i)
        _, predictions = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optim[i].step()

        inputs_representation = inputs_representation.detach()

    return loss, predictions


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


def perform_train_step(model, inputs, labels, criterion, optim, is_dgl, is_cdni):
    """
    Perform a single train-step, which is done differently when using regular training, DGL and cDNI.

    :param model: The model.
    :param inputs: The inputs.
    :param labels: The labels.
    :param criterion: The criterion.
    :param optim: The optimizer (or plural optimizers).
    :param is_dgl: Whether this model is trained using DGL (affects the optimizer and the train-step functionality).
    :param is_cdni: Whether this model is trained using DNI with context or not (affects the train-step functionality).
    :return: The loss of this train-step, as well as the predictions.
    """
    if is_dgl:
        loss, predictions = perform_train_step_dgl(model, inputs, labels, criterion, optim)
    elif is_cdni:
        loss, predictions = perform_train_step_cdni(model, inputs, labels, criterion, optim)
    else:
        loss, predictions = perform_train_step_regular(model, inputs, labels, criterion, optim)

    return loss, predictions


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
            parameters_to_train = itertools.chain(model.blocks[i].parameters(), model.auxiliary_nets[i].parameters())
            optimizer = optimizer_constuctor(parameters_to_train, **optimizer_params)
            optimizers.append(optimizer)

        return optimizers
    else:
        return optimizer_constuctor(model.parameters(), **optimizer_params)


def train_model(model, criterion, optimizer_params, dataloaders, dataset_sizes, device,
                num_epochs=25, log_interval=100, is_dgl=False, is_cdni=False):
    """
    A general function to train a model and return the best model found.

    :param model: the model to train
    :param criterion: which loss to train on
    :param optimizer_params: the optimizer to train with
    :param dataloaders: the dataloaders to feed the model
    :param dataset_sizes: the sizes of the datasets
    :param device: which device to train on
    :param num_epochs: how many epochs to train
    :param log_interval: How many training/testing steps between each logging (to wandb).
    :param is_dgl: Whether this model is trained using DGL (affects the optimizer and the train-step functionality).
    :param is_cdni: Whether this model is trained using DNI with context or not (affects the train-step functionality).
    :return: the model with the lowest test error
    """
    begin_time = time.time()

    best_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    optim: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]] = get_optim(model, optimizer_params, is_dgl)

    training_step = 0  # Will be incremented at the beginning of each training-step.

    # Accumulate loss and correct predictions of the log interval, to calculate later interval loss & accuracy.
    interval_loss_sum = 0.0
    interval_corrects_sum = 0

    for epoch in range(num_epochs):
        epoch_begin_time = time.time()
        model.train()

        # Accumulate loss and correct predictions of the entire epoch, to calculate later epoch loss & accuracy.
        epoch_loss_sum = 0.0
        epoch_corrects_sum = 0

        for batch_index, (inputs, labels) in enumerate(dataloaders['train']):
            training_step += 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            loss, predictions = perform_train_step(model, inputs, labels, criterion, optim, is_dgl, is_cdni)

            minibatch_size = inputs.size(0)  # This equals the batch-size argument except in the last mini-batch

            current_loss_value = loss.item()
            current_corrects_count = torch.sum(torch.eq(predictions, labels.data)).item()

            epoch_loss_sum += current_loss_value * minibatch_size
            epoch_corrects_sum += current_corrects_count
            interval_loss_sum += current_loss_value
            interval_corrects_sum += current_corrects_count

            if training_step % log_interval == 0:
                interval_loss = interval_loss_sum / log_interval
                interval_accuracy = (100 * interval_corrects_sum) / (log_interval * minibatch_size)
                wandb.log(data={'train_accuracy': interval_accuracy, 'train_loss': interval_loss}, step=training_step)
                # logger.debug(f'Train-step {training_step:0>8d} | '
                #              f'loss={interval_loss:.4f} accuracy={interval_accuracy:.2f}')

                interval_loss_sum = 0.0
                interval_corrects_sum = 0

        epoch_train_loss = epoch_loss_sum / dataset_sizes['train']
        epoch_train_accuracy = 100 * (epoch_corrects_sum / dataset_sizes['train'])

        epoch_test_loss, epoch_test_accuracy = evaluate_model(model, criterion, dataloaders['test'], device)
        wandb.log(data={'test_accuracy': epoch_test_accuracy, 'test_loss': epoch_test_loss}, step=training_step)

        # if the current model reached the best results so far,
        # deep copy the weights of the model
        if epoch_test_accuracy > best_accuracy:
            best_accuracy = epoch_test_accuracy
            best_weights = copy.deepcopy(model.state_dict())

        epoch_time_elapsed = time.time() - epoch_begin_time
        logger.info(f'Epoch {epoch+1:0>2d}/{num_epochs:0>2d} | '
                    f'{int(epoch_time_elapsed // 60):d}m {int(epoch_time_elapsed) % 60:0>2d}s | '
                    f'Train loss={epoch_train_loss:.4f} accuracy={epoch_train_accuracy:.2f}% | '
                    f'Test loss={epoch_test_loss:.4f} accuracy={epoch_test_accuracy:.2f}%')

    time_elapsed = time.time() - begin_time
    logger.info(f'Training {num_epochs:0>2d} epochs completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best test accuracy: {best_accuracy:.2f}%')

    # load best model weights
    model.load_state_dict(best_weights)
    return model
