import torch
import numpy as np

import os

import matplotlib.pyplot as plt


BASE_FREQS = [2 ** i for i in range(5)]

def generate_basis(num_points, func_range = (0, 10)):
    #generate basis sinusoids
    basis = []
    for freq in BASE_FREQS:
        curr_basis = np.sin(np.linspace(func_range[0], func_range[1], num_points) * freq)
        basis.append(curr_basis)
    return basis
if __name__ == '__main__':
    num_basis = len(BASE_FREQS)
    basis = generate_basis(1000)
    # Generate a random signal with sinusoidal components at the base frequencies at random amplitudes
    coeffs = np.random.rand(num_basis)
    signal = np.zeros_like(basis[0])
    for i in range(num_basis):
        signal += coeffs[i] * basis[i]

    # Plot the signal
    x = np.linspace(0, 10, 1000)
    plt.plot(x, signal)
    plt.show()
    a = 0




