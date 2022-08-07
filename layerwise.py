import torch

from pathlib import Path
from typing import List, Tuple

import wandb
from torch import nn
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
                   get_cnn)
from utils import initialize_model as initialize_model_trained_end2end


class LayerwiseVGG(LitVGG):
    """A variant of the VGG model that uses layerwise optimization."""

    def __init__(self, args: Args):
        super().__init__(args.arch, args.opt, args.data)
        self.aux_nets = nn.ModuleList(self.get_auxiliary_networks(args.layerwise))

    def get_auxiliary_networks(self, args: LayerwiseArgs) -> List[nn.Module]:
        """Get the auxiliary networks predicting the target classes' scores.

        For each intermediate module in the neural-network, create an auxiliary network which gets the output
        of that intermediate module as an input, and predict the target classes' scores.
        These auxiliary networks will be used eventually to propagate loss to the network's modules.

        Returns:
            A list of auxiliary networks.
        """
        aux_nets = list()
        for i in range(len(self.features) - 1):
            channels, height, width = self.shapes[i + 1]
            assert height == width, "Only square tensors are supported"
            spatial_size = height

            if args.pred_aux_type == 'mlp':
                aux_net = get_mlp(input_dim=channels * height * width,
                                  output_dim=self.data_args.n_classes,
                                  n_hidden_layers=args.aux_mlp_n_hidden_layers,
                                  hidden_dimensions=args.aux_mlp_hidden_dim)
            else:  # args.pred_aux_type == 'cnn'
                aux_net = get_cnn(conv_channels=[channels],
                                  linear_channels=[args.aux_mlp_hidden_dim] * args.aux_mlp_n_hidden_layers,
                                  in_spatial_size=spatial_size, in_channels=channels,
                                  n_classes=self.data_args.n_classes)
            aux_nets.append(aux_net)

        aux_nets.append(self.mlp)  # The "auxiliary network" used to train the last conv block is the network's mlp

        return aux_nets

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
        inputs, labels = batch
        intermediate_losses = list()

        x = inputs.clone()
        for i in range(len(self.features)):
            x = self.features[i](x)
            logits = self.aux_nets[i](x)
            loss = self.loss(logits, labels)
            predictions = torch.argmax(logits, dim=1)
            accuracy = torch.sum(labels == predictions).item() / len(labels)

            prefix = f'{stage.value}'
            if i < len(self.features) - 1:
                prefix += f'_module_{i}'

            self.log(f'{prefix}_loss', loss)
            self.log(f'{prefix}_accuracy', accuracy, on_epoch=True, on_step=False)

            intermediate_losses.append(loss)
            x = x.detach()

        return sum(intermediate_losses)


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
    datamodule = initialize_datamodule(args.data, args.opt.batch_size)
    wandb_logger = initialize_wandb_logger(args)
    configure_logger(args.env.path, level='DEBUG')
    model = initialize_model(args, wandb_logger)

    wandb.watch(model, log='all')

    trainer = initialize_trainer(args.env, args.opt, wandb_logger)
    trainer.fit(model, datamodule=datamodule)
    unwatch_model(model)


if __name__ == '__main__':
    main()
