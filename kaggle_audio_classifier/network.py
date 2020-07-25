import torch
import torch.nn as nn
from util import Flatten
def discriminator(opts, output_size):
    """
    From https://arxiv.org/abs/1511.06434.pdf
    """
    model = nn.Sequential(
        nn.Conv2d(1, 32, 4, 2, 1), #1
        nn.InstanceNorm2d(32),
        nn.LeakyReLU(0.2, True),
        nn.Dropout3d(opts.dropout),
        nn.Conv2d(32, 32, 4, 2, 1), #2
        nn.InstanceNorm2d(32),
        nn.LeakyReLU(0.2, True),
        nn.Dropout3d(opts.dropout),
        nn.Conv2d(32, 64, 4, 2, 1), #3
        nn.InstanceNorm2d(64),
        nn.LeakyReLU(0.2, True),
        nn.Dropout3d(opts.dropout),
        nn.Conv2d(64, 64, 4, 2, 1), #4
        nn.InstanceNorm2d(64),
        nn.LeakyReLU(0.2, True),
        nn.Dropout3d(opts.dropout),
        nn.Conv2d(64, 128, 4, 2, 1), #5
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(0.2, True),
        nn.Dropout3d(opts.dropout),
        nn.Conv2d(128, 128, 5,1, 1),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(0.2, True),
        nn.Dropout3d(opts.dropout),

        Flatten(),
        nn.Linear(128 * 29, 32 * 29),
        nn.LeakyReLU(0.2, True),
        nn.Dropout(opts.dropout),
        nn.Linear(32 * 29, 16 * 29),
        nn.LeakyReLU(0.2, True),
        nn.Dropout(opts.dropout),
        nn.Linear(16 * 29, output_size),
    )

    return model