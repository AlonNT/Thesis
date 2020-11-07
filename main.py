from time import strftime

import json
import tqdm
import loguru
import argparse
import torch.nn as nn
import torch.nn.functional as functional


class SimpleConvNet(nn.Module):
    """
    A simple Convolutional Neural Network.
    The architecture is:
    Conv-Relu-Pool-Conv-Relu-Pool-Affn-Relu-Affn-Relu-Affn
    """

    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    args = parse_args()
    curr_time = strftime("%Y-%m-%d_%H-%M-%S")
    tqdm.write("Started at {}".format(curr_time))

    # If CUDA is available use the given device_num.
    # If not - use CPU.
    device = torch.device("cuda:{}".format(device_num)
                          if torch.cuda.is_available() else "cpu")
    tqdm.write("Using device: {}".format(device))

    # Set arguments with their default values. Since it's a list which is mutable,
    # it needs to be constructed this way and not in the function definition.
    if bs is None:
        bs = [32]
    if lr is None:
        lr = [0.1]
    if momentum is None:
        momentum = [0.9]
    if weight_decay is None:
        weight_decay = [0.001]

    # Initialize the embedding as the given option.
    if init_embedding_as == 'RGB':
        initial_embedding = torch.tensor(create_rgb(embedding_size), device=device)
        tqdm.write("Initializing the embedding layer with RGB.")




def parse_args():
    parser = argparse.ArgumentParser(
        description='Beyond Gradient-Descent main script'
    )

    parser.add_argument('--lr', default=[0.1], type=float, nargs='+',
                        help='[default 0.1]')
    parser.add_argument('--bs', default=[32], type=int, nargs='+',
                        help='[default 32]')
    parser.add_argument('--momentum', default=[0.9], type=float, nargs='+',
                        help='[default 0.9]')
    parser.add_argument('--weight_decay', default=[0], type=float, nargs='+',
                        help='[default 0, which means no weight-decay]')
    parser.add_argument('--epochs', default=5, type=int,
                        help='number of epochs [default 5].')

    parser.add_argument('--use_dni', action='store_true',
                        help='If indicated - shuffle the colors of the images given to the training')

    parser.add_argument('--device_num', type=int, default=0,
                        help='which device to train on (will be used if CUDA is available)')
    parser.add_argument('--verbose', action='store_true',
                        help='progress bar and extra information')
    parser.add_argument('--save_model', action='store_true',
                        help="If indicated, the model with the best test accuracy will be saved.")

    return parser.parse_args()


if __name__ == '__main__':
    main()
