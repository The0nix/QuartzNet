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
                nn.Conv1d(C_in if i == 0 else C, C, # C_in inputs for first, C for others
                          kernel_size=K,
                          groups=C_in if i == 0 else C,
                          padding=(K - 1) // 2),
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
                if not (i == len(self.blocks) - 1 and j == len(block) -1 ):  # If not last ReLU
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
    :param S: iterable of 5 values designating repetitions of each B_i block or integer if all repetitions are the same
    :param n_labels: number of output labels
    """
    def __init__(self, C_in, Ss: Union[Collection, int], n_labels) -> None:
        super().__init__()
        assert isinstance(Ss, int) or len(Ss) == 5
        if isinstance(Ss, int):
            Ss = [Ss] * 5

        # TODO: put all parameters into some config
        Cs = [256, 256, 512, 512, 512]  # Output channels in blocks
        Ks = [33, 39, 51, 63, 75]       # Kernel sizes in blocks
        Rs = [5, 5, 5, 5, 5]            # Number of repetitions inside of each block

        self.C1 = nn.Sequential(
            nn.Conv1d(C_in, 256, kernel_size=33, padding=(33 - 1) // 2),
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
            nn.Conv1d(512, 512, kernel_size=87, padding=(87 - 1) // 2),
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
        return x.log_softmax(axis=1)
