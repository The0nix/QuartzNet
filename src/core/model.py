from typing import Union, Collection

import torch.nn as nn


class QuartzBlock(nn.Module):
    """
    Basic block of QuartzNet consisting of Separable Convolution, BatchNorm and ReLU repeating R times
    :
    :param C_in: number of input channels
    :param C: C from paper (Channels) -- number of output channels
    :param K: K from paper (Kernel) -- size of kernels
    :param R: R from paper (Repeats) -- number of repetitions of block constituents
    """
    def __init__(self, C_in: int, C: int, K: int, R: int) -> None:
        super().__init__()
        self.R = R
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                # TODO: Is this really "time-channel separable conv"?
                nn.Conv1d(C_in if i == 0 else C, C,      # C_in inputs for first, C for others
                          kernel_size=K,
                          groups=C_in if i == 0 else C,  # same
                          padding=K // 2),
                nn.Conv1d(C, C, kernel_size=1),
                nn.BatchNorm1d(C),
                nn.ReLU(),
            ])
            for i in range(R)
        ])
        self.res_conv = nn.Sequential(  # convolution for residual
            nn.Conv1d(C_in, C, kernel_size=1),
            nn.BatchNorm1d(C),
        )

    def forward(self, x):
        x_initial = x
        for i, block in enumerate(self.blocks):
            for j, layer in enumerate(block):
                if not (i == len(self.blocks) - 1 and j == len(block) - 1):  # If not last ReLU
                    x = layer(x)
                else:
                    # Pass residual
                    x = x + self.res_conv(x_initial)
                    x = layer(x)
        return x


class QuartzNet(nn.Module):
    """
    QuartzNet ASR model combining QuartzBlocks and CTC
    :param C_in: number of input channels
    :param Ss: iterable of 5 values designating repetitions of each B_i block or integer if all repetitions are the same
    :param Cs: Output channels in blocks
    :param Ks: Kernel sizes in blocks
    :param Rs: Number of repetitions inside of each block
    :param n_labels: number of output labels
    """
    def __init__(self, C_in, n_labels: int, Cs: Collection, Ks: Collection, Rs: Collection,
                 Ss: Union[Collection, int]) -> None:
        super().__init__()
        assert isinstance(Ss, int) or len(Ss) == 5, "Ss must be an int or collection of length 5"
        assert len(Cs) == 5, "Cs must be a collection of length 5"
        assert len(Ks) == 5, "Cs must be a collection of length 5"
        assert len(Rs) == 5, "Cs must be a collection of length 5"
        if isinstance(Ss, int):
            Ss = [Ss] * 5
        self.n_labels = n_labels

        self.C1 = nn.Sequential(
            nn.Conv1d(C_in, 256, kernel_size=33, padding=33 // 2, stride=2),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.Bs = nn.Sequential(
            *[
                QuartzBlock(C_in_ if i == 0 else C, C, K, R)  # C_in if first out of S repetitions else C
                for C_in_, C, K, R, S in zip([256] + Cs[:-1], Cs, Ks, Rs, Ss)
                for i in range(S)  # Repeat each QuartzBlock S times
            ]
        )

        self.C2 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=87, dilation=2, padding=87 - 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.C3 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.C4 = nn.Sequential(
            nn.Conv1d(1024, n_labels, kernel_size=1),
        )

    def forward(self, x):
        x = self.C1(x)
        x = self.Bs(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        return x.log_softmax(dim=1)
