from dataclasses import dataclass, field

import abc
import os

from typing import List

@dataclass
class BaseOptions(abc.ABC):
    exp_dir: str = './exps/'
    device: str = 'cuda'

    @abc.abstractmethod
    def model_name(self):
        pass

@dataclass
class PixelNeRF1DOptions(BaseOptions):
    #data options
    num_points: int = 1000
    num_freqs: int = 3
    num_funcs: int = 1
    #convnet options
    num_features: int = 8
    conv_depth: int = 6
    kernel_size: int = 5


    #model options
    pe_res: int = 2

    #training options
    lr: float = 5e-4
    num_steps = 5000
    def model_name(self):
        return 'pixel_nerf_1d'
