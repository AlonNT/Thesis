import numpy as np
import torch
import torchvision

from pathlib import Path
from typing import List, Tuple, Optional

import wandb
from torch import nn
from torchvision.transforms.functional import resize
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.states import RunningStage

from schemas.layerwise import Args, LayerwiseArgs
from utils import (configure_logger,
                   get_args,
                   initialize_datamodule,
                   initialize_wandb_logger,
                   initialize_trainer,
                   unwatch_model,
                   LitVGG,
                   get_mlp,
                   get_cnn,
                   log_args)
from utils import initialize_model as initialize_model_trained_end2end


class LayerwiseVGG(LitVGG):
    """A variant of the VGG model that uses layerwise optimization."""

    def __init__(self, args: Args):
        super().__init__(args.arch, args.opt, args.data)
        self.layerwise_args: LayerwiseArgs = args.layerwise
        self.reconstruction_loss = nn.L1Loss() if self.reconstruction_on else None  # TODO: Consider trying L2Loss

        self.classification_auxiliary_networks = self.init_classification_auxiliary_networks()
        self.reconstruction_auxiliary_networks = self.init_reconstruction_auxiliary_networks()

    def init_classification_auxiliary_networks(self) -> Optional[nn.ModuleList]:
        """Get the auxiliary networks predicting the target classes' scores.

        For each intermediate module in the neural-network, create an auxiliary network which gets the output
        of that intermediate module as an input, and predict the target classes' scores.
        These auxiliary networks will be used eventually to propagate loss to the network's modules.

        Returns:
            A list of auxiliary networks.
        """
        if not self.classification_on:
            return None

        args = self.layerwise_args
        classification_auxiliary_networks = list()

        for i in range(len(self.features) - 1):
            channels, height, width = self.shapes[i + 1]
            assert height == width, "Only square tensors are supported"
            spatial_size = height

            if args.classification_aux_type == 'mlp':
                auxiliary_network = get_mlp(input_dim=channels * height * width,
                                            output_dim=self.data_args.n_classes,
                                            n_hidden_layers=args.aux_mlp_n_hidden_layers,
                                            hidden_dimensions=args.aux_mlp_hidden_dim)
            else:  # args.pred_aux_type == 'cnn'
                auxiliary_network = get_cnn(conv_channels=[channels],
                                            linear_channels=[args.aux_mlp_hidden_dim] * args.aux_mlp_n_hidden_layers,
                                            in_spatial_size=spatial_size, in_channels=channels,
                                            n_classes=self.data_args.n_classes)

            classification_auxiliary_networks.append(auxiliary_network)

        # The "auxiliary network" used to train the last conv block is the network's mlp
        classification_auxiliary_networks.append(self.mlp)

        return nn.ModuleList(classification_auxiliary_networks)

    def init_reconstruction_auxiliary_networks(self) -> Optional[nn.ModuleList]:
        """Initializes the decoders (predicting the original input image) from the different intermediate tensors.

        These decoders essentially make every subnetwork (layers 1->...->i) an auto-encoder.
        The gradients propagated from these decoders are used to update the intermediate layers,
        therefore enforcing them to learn "good" features, which hopefully benefit downstream layers.

        Returns:
            A list of nn.Sequential modules, each is a decoder.
        """
        if not self.reconstruction_on:
            return None

        args = self.layerwise_args

        reconstruction_auxiliary_networks = list()
        for i in range(len(self.features)):
            in_channels, height, width = self.shapes[i + 1]
            assert height == width, "Only square tensors are supported"
            spatial_size = height
            upsampling_kwargs = dict(desired_out_channels=self.data_args.n_channels,
                                     in_spatial_size=spatial_size,
                                     out_spatial_size=self.data_args.spatial_size) if args.upsample else dict()
            auxiliary_network = get_auto_decoder(in_channels, **upsampling_kwargs)
            reconstruction_auxiliary_networks.append(auxiliary_network)

        return nn.ModuleList(reconstruction_auxiliary_networks)

    def shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: RunningStage):
        """Performs train/validation step, depending on the given `stage`.

        Each convolution block in the network is trained separately, using the loss of the auxiliary network.
        Technically, this is implemented by defining the loss of the whole network
        to be the sum of the losses of the auxiliary networks.
        Note that after each convolution block propagates its input tensor,
        the tensor is detached to prevent backpropagation to the preceding blocks.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            stage: Indicating if this is a training-step or a validation-step.

        Returns:
            The loss.
        """
        args = self.layerwise_args

        inputs, labels = batch
        classification_losses = list()
        reconstruction_losses = list()

        x = inputs.clone()
        for i in range(len(self.features)):
            x = self.features[i](x)

            prefix = f'{stage.value}' if (i == len(self.features) - 1) else f'{stage.value}_module_{i}'

            if self.classification_on:
                classification_losses.append(self.get_classification_loss(x, labels, i, prefix))
            if self.reconstruction_on:
                reconstruction_losses.append(self.get_reconstruction_loss(x, inputs, i, prefix))

            x = x.detach()

        loss = 0

        if self.classification_on:
            classification_loss = sum(classification_losses)
            loss += args.classification_loss_weight * classification_loss
        else:
            assert self.reconstruction_on and (args.mlp_loss_weight > 0), \
                'When training without classification loss, the modules should be trained with reconstruction loss ' \
                'for the convolution modules, and the MLP should be trained (i.e. its loss weight should be positive).'
            # When training with reconstruction loss only, the MLP is trained on the features of the last conv block.
            # Note that this part does not propagate gradients to the network's convolution modules, only to the mlp.
            mlp_loss = self.get_classification_loss(x, labels, i=len(self.features) - 1, prefix=f'{stage.value}')
            loss += args.mlp_loss_weight * mlp_loss
        if self.reconstruction_on:
            reconstruction_loss = sum(reconstruction_losses)
            loss += args.reconstruction_loss_weight * reconstruction_loss

        return loss

    @property
    def reconstruction_on(self) -> bool:
        return self.layerwise_args.reconstruction_loss_weight > 0

    @property
    def classification_on(self) -> bool:
        return self.layerwise_args.classification_loss_weight > 0

    @property
    def shifts(self) -> Optional[tuple[int]]:
        return tuple(np.linspace(start=self.kernel_sizes[0],
                                 stop=self.data_args.spatial_size // 2,
                                 num=len(self.features),
                                 dtype=int)) if self.layerwise_args.shift_ssl_labels else None

    def get_classification_loss(self, x: torch.Tensor, labels: torch.Tensor, i: int, prefix: str) -> torch.Tensor:
        logits = self.classification_auxiliary_networks[i](x)
        classification_loss = self.loss(logits, labels)
        predictions = torch.argmax(logits, dim=1)
        accuracy = torch.sum(labels == predictions).item() / len(labels)

        self.log(f'{prefix}_loss', classification_loss)
        self.log(f'{prefix}_accuracy', accuracy, on_epoch=True, on_step=False)

        return classification_loss

    def get_reconstruction_loss(self, x: torch.Tensor, inputs: torch.Tensor, i: int, prefix: str) -> torch.Tensor:
        reconstructed_inputs = self.reconstruction_auxiliary_networks[i](x)
        reconstruction_labels = inputs.detach()
        reconstructed_input_spatial_size = reconstructed_inputs.shape[-1]
        if reconstructed_input_spatial_size != self.data_args.spatial_size:
            assert not self.layerwise_args.upsample, \
                'If `upsample` was given, the reconstructed inputs should be the same size as the inputs.'
            assert reconstructed_input_spatial_size < self.data_args.spatial_size, \
                'The reconstructed inputs should be smaller than the original inputs.'
            reconstruction_labels = resize(reconstruction_labels, size=[reconstructed_input_spatial_size] * 2)

        if self.layerwise_args.shift_ssl_labels:
            assert self.layerwise_args.upsample, \
                'When shifting reconstruction labels, it makes more sense to upsample during reconstruction'
            reconstruction_labels = torch.roll(reconstruction_labels, shifts=2 * (self.shifts[i],), dims=(2, 3))

        reconstruction_loss = self.reconstruction_loss(reconstructed_inputs, reconstruction_labels)
        self.log(f'{prefix}_reconstruction_loss', reconstruction_loss)

        indices_to_plot = np.random.choice(inputs.size(0), size=4, replace=False)
        images = torch.cat([reconstruction_labels[indices_to_plot], reconstructed_inputs[indices_to_plot].detach()])
        grid = torchvision.utils.make_grid(images, nrow=4)
        wandb_image = wandb.Image(grid.cpu().numpy().transpose((1, 2, 0)))
        self.logger.experiment.add_image(f'reconstructions-#{i}', wandb_image, 0)

        return reconstruction_loss


def get_auto_decoder(in_channels: int,
                     desired_out_channels: int = 3,
                     in_spatial_size: Optional[int] = None,
                     out_spatial_size: Optional[int] = None) -> nn.Sequential:
    """Gets an auxiliary-network predicting the original input image from the given tensor.

    This network predicts an image, i.e. a tensor of shape `target_image_size` x `target_image_size` x 3,
    given an input tensor of size `image_size` x `image_size` x `channels`
    If `image_size` and `target_image_size` are given, the aux-net first upsample using a stack of ConvTranspose2d.
    Each up-sampling is done by doubling the spatial size and halving the number of channels.

    Args:
        in_channels: Number of channels in the input tensor of the returned aux-net.
        desired_out_channels: Number of channels in the output tensor of the returned aux-net.
        in_spatial_size: If given, this is the source image size (to be upsampled).
        out_spatial_size: If given, this is the target image size (to be upsampled to).
    Returns:
        The auxiliary network.
    """
    layers: List[nn.Module] = list()

    assert (in_spatial_size is None) == (out_spatial_size is None), \
        "image_size and target_image_size should be both None (in which case no up-sampling is done), " \
        "or both not None (in which case upsampling is done from image_size to target_image_size)."

    if in_spatial_size is not None:
        while in_spatial_size != out_spatial_size:
            out_channels = in_channels // 2

            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())

            in_spatial_size *= 2
            in_channels = out_channels

    # Convolution layer with 1x1 kernel to change channels to out_channels.
    layers.append(nn.Conv2d(in_channels, desired_out_channels, kernel_size=1))

    return nn.Sequential(*layers)


def initialize_model_trained_layerwise(args: Args, wandb_logger: WandbLogger):
    if args.arch.model_name.startswith('VGG'):
        model_class = LayerwiseVGG
    elif any(args.arch.model_name.startswith(s) for s in ['D-', 'S-']):
        raise NotImplementedError("Basic CNN-style networks are not implemented layerwise yet.")
    else:
        raise NotImplementedError("MLP networks are not implemented layerwise yet.")

    if args.arch.use_pretrained:
        artifact = wandb_logger.experiment.use_artifact(args.arch.pretrained_path, type='model')
        artifact_dir = artifact.download()
        model = model_class.load_from_checkpoint(str(Path(artifact_dir) / "model.ckpt"), args=args)
    else:
        model = model_class(args)

    return model


def initialize_model(args: Args, wandb_logger: WandbLogger):
    if args.layerwise.dgl:
        return initialize_model_trained_layerwise(args, wandb_logger)
    else:
        return initialize_model_trained_end2end(args.arch, args.opt, args.data, wandb_logger)


def main():
    args = get_args(args_class=Args)
    configure_logger(args.env.path, level='DEBUG')
    log_args(args)

    datamodule = initialize_datamodule(args.data, args.opt.batch_size)
    wandb_logger = initialize_wandb_logger(args)
    model = initialize_model(args, wandb_logger)
    wandb.watch(model, log='all')

    trainer = initialize_trainer(args.env, args.opt, wandb_logger)
    trainer.fit(model, datamodule=datamodule)
    unwatch_model(model)


if __name__ == '__main__':
    main()
