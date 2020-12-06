import copy
import time

import torch
import torchvision
import itertools
import wandb

from typing import List
from loguru import logger


def get_data(batch_size=32):
    """
    Get the CIFAR10 data.
    :param batch_size: the size of the mini-batches to initialize the dataloaders
    :return: image_datasets: dictionary mapping "train"/"test" to its dataset
             dataloaders:    dictionary mapping "train"/"test" to its dataloader
             dataset_sizes:  dictionary mapping "train"/"test" to its dataset size
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
                                                      download=False)
                      for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=0)
                   for x in ['train', 'test']}

    return image_datasets, dataloaders, dataset_sizes


def evaluate_model(model: torch.nn.Module, criterion, dataloader, device):
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
            if isinstance(result, tuple):
                _, outputs = result
            else:
                outputs = result

            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        loss_sum += loss.item() * inputs.size(0)
        corrects_sum += torch.sum(torch.eq(predictions, labels.data)).item()

    loss = loss_sum / len(dataloader.dataset)
    accuracy = 100 * (corrects_sum / len(dataloader.dataset))

    return loss, accuracy


def perform_train_step_dgl(model, inputs, labels, criterion, optim):
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


def perform_train_step_regular(model, inputs, labels, criterion, optim):
    optim.zero_grad()

    outputs = model(inputs)
    _, predictions = torch.max(outputs, 1)
    loss = criterion(outputs, labels)

    loss.backward()
    optim.step()

    return loss, predictions


def perform_train_step(model, inputs, labels, criterion, optim, device, is_dgl):
    inputs = inputs.to(device)
    labels = labels.to(device)

    if is_dgl:
        loss, predictions = perform_train_step_dgl(model, inputs, labels, criterion, optim)
    else:
        loss, predictions = perform_train_step_regular(model, inputs, labels, criterion, optim)

    return loss, predictions


def get_optim(model, optimizer_params, is_dgl):
    if is_dgl:
        optimizers: List[torch.optim.Optimizer] = list()

        for i in range(len(model.blocks)):
            parameters_to_train = itertools.chain(model.blocks[i].parameters(), model.auxiliary_nets[i].parameters())
            optimizer: torch.optim.Optimizer = torch.optim.SGD(parameters_to_train, **optimizer_params)
            optimizers.append(optimizer)

        return optimizers
    else:
        return torch.optim.SGD(model.parameters(), **optimizer_params)


def train_model(model, criterion, optimizer_params, dataloaders, dataset_sizes, device,
                num_epochs=25, log_interval=100, is_dgl=False):
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
    :param is_dgl: Whether this model is trained using Decoupled-Greedy-Learning.
    :return: the model with the lowest test error
    """
    since = time.time()

    best_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    optim: List[torch.optim.Optimizer] = get_optim(model, optimizer_params, is_dgl)

    training_step = 0  # Will be incremented at the beginning of each training-step

    # Accumulate loss and correct predictions of the log interval, to calculate later interval loss & accuracy.
    interval_loss_sum = 0.0
    interval_corrects_sum = 0

    for epoch in range(num_epochs):
        epoch_begin_time = time.time()
        model.train()

        # Accumulate loss and correct predictions of the entire epoch, to calculate later epoch loss & accuracy.
        epoch_loss_sum = 0.0
        epoch_corrects_sum = 0

        for inputs, labels in dataloaders['train']:
            training_step += 1
            loss, predictions = perform_train_step(model, inputs, labels, criterion, optim, device, is_dgl)

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

    time_elapsed = time.time() - since
    logger.info(f'Training {num_epochs:0>2d} epochs completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best test accuracy: {100 * best_accuracy:.2f}%')

    # load best model weights
    model.load_state_dict(best_weights)
    return model
