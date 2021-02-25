import torch


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

    return loss.item(), predictions
