import copy
import time

import torch
import torchvision

from loguru import logger


def get_data(bs=32):
    """
    Get the CIFAR10 data.
    :param bs: the size of the minibatches to initialize the dataloaders
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
                                                  batch_size=bs,
                                                  shuffle=True,
                                                  num_workers=0)
                   for x in ['train', 'test']}

    return image_datasets, dataloaders, dataset_sizes


def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=25):
    """
    A general function to train a model and return the best model found.

    :param model: the model to train
    :param criterion: which loss to train on
    :param optimizer: the optimizer to train with
    :param dataloaders: the dataloaders to feed the model
    :param dataset_sizes: the sizes of the datasets
    :param device: which device to train on
    :param num_epochs: how many epochs to train
    :return: the model with the lowest test error
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        evaluation = {'train': dict(), 'test': dict()}
        epoch_begin_time = time.time()
        for phase in ['train', 'test']:
            is_training = (phase == 'train')
            model.train(mode=is_training)

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(is_training):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if is_training:
                        loss.backward()
                        optimizer.step()

                    # for i, l in enumerate(model.layers):
                    #     if isinstance(l, torch.nn.Conv2d) or isinstance(l, torch.nn.Linear):
                    #         logger.debug(f'layer #{i} | '
                    #                      f'min ={model.layers[i].weight.grad.detach().numpy().min():.2f}, '
                    #                      f'mean={model.layers[i].weight.grad.detach().numpy().mean():.2f}, '
                    #                      f'max ={model.layers[i].weight.grad.detach().numpy().max():.2f}')

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            evaluation[phase]['loss'] = running_loss / dataset_sizes[phase]
            evaluation[phase]['acc'] = running_corrects.double() / dataset_sizes[phase]

            # if the current model reached the best results so far,
            # deep copy the weights of the model
            if (not is_training) and (evaluation[phase]['acc'] > best_acc):
                best_acc = evaluation[phase]['acc']
                best_model_wts = copy.deepcopy(model.state_dict())

        epoch_time_elapsed = time.time() - epoch_begin_time
        logger.info(f'Epoch {epoch+1:0>2d}/{num_epochs:0>2d} | '
                    f'{int(epoch_time_elapsed // 60):d}m {int(epoch_time_elapsed) % 60:0>2d}s | '
                    f'Train loss={evaluation["train"]["loss"]:.4f} accuracy={100 * evaluation["train"]["acc"]:.2f}% | '
                    f'Test loss={evaluation["test"]["loss"]:.4f} accuracy={100 * evaluation["test"]["acc"]:.2f}%')

    time_elapsed = time.time() - since
    logger.info(f'Training {num_epochs:0>2d} epochs completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best test accuracy: {100 * best_acc:.2f}%')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
