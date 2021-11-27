import torch
from utils import cross_entropy_gradient, get_optim, get_dataloaders, Accumulator, perform_train_step_direct_global
from vgg import VGGwDGL

from typing import Tuple


def test_cross_entropy_gradient():
    """
    Test the function cross_entropy_gradient by giving feeding random inputs and labels
    to a basic model consisting of a single linear layer followed by a cross-entropy loss.
    """
    n_input_channels = 3
    image_size = 32
    batch_size = 100
    n_classes = 10

    inputs = torch.rand(batch_size, n_input_channels, image_size, image_size)
    labels = torch.randint(low=0, high=n_classes, size=(batch_size,))

    flatten = torch.nn.Flatten()
    linear = torch.nn.Linear(in_features=n_input_channels * (image_size ** 2), out_features=n_classes)
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    logits = linear(flatten(inputs))
    loss = cross_entropy_loss(logits, labels)

    logits.retain_grad()
    loss.backward()

    logits_grad = (1 / batch_size) * cross_entropy_gradient(logits, labels)
    actual_grad = logits.grad

    assert torch.allclose(logits_grad,actual_grad), \
        f'\nCalculated grad:\n{logits_grad.detach().numpy()}\nActual grad:\n{actual_grad.detach().numpy()}'


def test_cross_entropy_gradient_in_vgg():
    """
    Test the function cross_entropy_gradient performing one train step in a VGG model,
    using hooks to validate the calculated gradient versus the actual one.
    """
    batch_size = 64
    dataloaders = get_dataloaders(batch_size)
    images, labels = next(iter(dataloaders['train']))

    def backward_hook(module: torch.nn.Module,
                      grad_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      grad_outputs: Tuple[torch.Tensor]):
        bias_grad, input_grad, weights_grad = grad_inputs
        output_grad, = grad_outputs
        my_output_grad = (1 / batch_size) * cross_entropy_gradient(module.forward_output_tensor, module.labels)

        assert torch.allclose(my_output_grad, output_grad), \
            f'FAILED: max-diff={torch.max(torch.abs(my_output_grad - output_grad)).item():.9f}'

    def forward_hook(module: torch.nn.Module,
                     inputs: Tuple[torch.Tensor],
                     output_tensor: torch.Tensor):
        input_tensor, = inputs
        module.register_buffer(name='forward_input_tensor', tensor=input_tensor, persistent=False)
        module.register_buffer(name='forward_output_tensor', tensor=output_tensor, persistent=False)
        module.register_buffer(name='labels', tensor=labels, persistent=False)

    # Create a model and register hooks to validate gradients calculation.
    model = VGGwDGL(vgg_name='VGG11c')
    for aux_net in model.auxiliary_nets:
        if aux_net is not None:
            aux_net_output_layer = aux_net[-1]
            aux_net_output_layer.register_backward_hook(backward_hook)
            aux_net_output_layer.register_forward_hook(forward_hook)

    perform_train_step_direct_global(model, images, labels, 
                                     criterion=torch.nn.CrossEntropyLoss(), 
                                     optimizers=get_optim(model, 
                                                          optimizer_params=dict(optimizer_type='SGD', 
                                                                                lr=1e-4, 
                                                                                weight_decay=0, 
                                                                                momentum=0.9), 
                                                          is_dgl=True),
                                     training_step=1, 
                                     modules_accumulators=[Accumulator() if (aux_net is not None) else None
                                                           for aux_net in model.auxiliary_nets],
                                     last_gradient_weight=0)
