import torch
import torch.nn as nn

import numpy as np

import os

import matplotlib.pyplot as plt

from dataset import PairDataset

from options import PixelNeRF1DOptions


class PE(nn.Module):
    def __init__(self, num_res = 6):
        super(PE, self).__init__()
        self.num_res = num_res
    def forward(self, x):
        outs = [x]
        for r in range(self.num_res):
            outs.append(torch.sin(x * 2 ** r))
            outs.append(torch.cos(x * 2 ** r))

        out = torch.cat(outs, dim=1)
        return out
class PriorCNN(nn.Module):
    def __init__(self, opts):
        super(PriorCNN, self).__init__()
        self.opts = opts

        self.num_features = self.opts.num_features

        conv_layers = [nn.Conv1d(in_channels=1, out_channels=self.num_features, kernel_size=self.opts.kernel_size, stride=1, padding=self.opts.kernel_size // 2)]
        for i in range(self.opts.conv_depth - 2):
            conv_layers.append(nn.Conv1d(in_channels=self.num_features, out_channels=self.num_features, kernel_size=self.opts.kernel_size, stride=1, padding=self.opts.kernel_size // 2))
            conv_layers.append(nn.BatchNorm1d(self.num_features))
            conv_layers.append(nn.ReLU())
        self.net = nn.Sequential(
            *conv_layers,
            nn.Conv1d(in_channels=self.num_features, out_channels=self.num_features, kernel_size=self.opts.kernel_size, stride=1, padding=self.opts.kernel_size // 2),

        )
    def forward(self, x):
        return self.net(x)

class PixelNeRFMLP(nn.Module):
    def __init__(self, opts):
        super(PixelNeRFMLP, self).__init__()

        self.opts = opts

        self.prior_conv = PriorCNN(self.opts)

        pos_size = 1
        self.use_pe = False
        if self.opts.pe_res > 0:
            self.pe = PE(self.opts.pe_res)
            pos_size = self.opts.pe_res * 2 + 1
            self.use_pe = True

        self.mlp = nn.Sequential(
            nn.Linear(self.opts.num_features + pos_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),

        )
    def forward(self, x, prior_in):
        #x: (B, N)
        #prior_in: (B, 1, N)
        if self.use_pe:
            x = self.pe(x)
        prior = self.prior_conv(prior_in)
        mlp_in = torch.cat([prior, x], dim = 1)
        B, C, L = mlp_in.shape
        mlp_in = mlp_in.permute(0,2, 1).reshape(B*L, C)

        return self.mlp(mlp_in)



if __name__ == '__main__':
    opts = PixelNeRF1DOptions()

    dataset = PairDataset(opts)

    model = PixelNeRFMLP(opts).to(opts.device)
    data, data_pair = dataset.__getitem__(0)

    data = data.unsqueeze(0).unsqueeze(0)

    x = torch.linspace(0, 10, opts.num_points).unsqueeze(0).unsqueeze(0).to(opts.device)
    out = model(x, data)

    a = 0
    #plot data
    # x = np.linspace(0, 10, opts.num_points)
    # plt.plot(x, dataset.data.cpu().numpy())
    # plt.plot(x, dataset.data_pair.cpu().numpy())
    # plt.show()

    a = 0