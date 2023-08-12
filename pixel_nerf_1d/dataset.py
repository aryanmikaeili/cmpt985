import torch
import numpy as np

import os

import matplotlib.pyplot as plt
from accelerate.utils import set_seed

from options import PixelNeRF1DOptions



class PairDataset(torch.utils.data.Dataset):
    def __init__(self, opts, set = 'train'):
        super(PairDataset, self).__init__()
        self.opts = opts
        self.set = set
        self.BASIS_FREQS = [2 ** i for i in range(self.opts.num_freqs)]
        self.num_points = self.opts.num_points
        self.basis = generate_basis(self.num_points, self.BASIS_FREQS, func_range = (0, 10))
        self.basis_diff = generate_diff_basis(self.num_points, self.BASIS_FREQS, func_range = (0, 10))
        self.data_list, self.data_pair_list = self.generate_data_pair()


    def __len__(self):
        return self.opts.num_funcs if self.set == 'train' else 1
    def __getitem__(self, idx):
        return torch.tensor(self.data_list[idx]).float().to(self.opts.device), torch.tensor(self.data_pair_list[idx]).float().to(self.opts.device)


    def generate_data_pair(self):
        coeffs = (np.random.rand(self.opts.num_freqs) * 3) + 1
        coeffs_pair = coeffs * 2

        #generate data
        data_list = []
        data_pair_list = []

        num_funcs = self.opts.num_funcs if self.set == 'train' else 1
        for i in range(num_funcs):
            data = np.sum([coeffs[i] * self.basis[i] for i in range(self.opts.num_freqs)], axis = 0)
            data_diff = np.sum([coeffs[i] * (2 ** i) *  self.basis_diff[i] for i in range(self.opts.num_freqs)], axis = 0) / 10.
            #data_diff = np.convolve(data, np.ones(3) / 3, mode = 'same') / 3.
            data_pair = np.sum([coeffs_pair[i] * self.basis[i] for i in range(self.opts.num_freqs)], axis = 0)

            data_pair *= data_diff


            data_list.append(data)
            data_pair_list.append(data_pair)
        return data_list, data_pair_list
def generate_basis(num_points, base_freqs, func_range = (0, 10)):
    #generate basis sinusoids
    basis = []
    for freq in base_freqs:
        curr_basis = np.sin(np.linspace(func_range[0], func_range[1], num_points) * freq)
        basis.append(curr_basis)
    return basis

def generate_diff_basis(num_points, base_freqs, func_range = (0, 10)):
    basis = []
    for freq in base_freqs:
        curr_basis = np.cos(np.linspace(func_range[0], func_range[1], num_points) * freq)
        basis.append(curr_basis)
    return basis
if __name__ == '__main__':
    set_seed(0)
    opts = PixelNeRF1DOptions()

    dataset = PairDataset(opts)

    #plot data
    x = np.linspace(0, 10, opts.num_points)
    plt.plot(x, dataset.data_list[0], label = 'prior')
    #`plt.plot(x, dataset.data_pair_list[0], label = 'unknown')
    plt.legend()
    plt.show()





