
import torch
import torch.nn as nn
from functools import reduce
from operator import mul
from .helper_classes import FlatImageForwardReshape
from efficient_kan.kan import KANLinear  

class KAN_Generator(nn.Module, FlatImageForwardReshape):
    def __init__(self, noise_dim, img_dim, 
                 grid_size=5, spline_order=3, 
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0):
        
        super().__init__()
        self.noise_dim = noise_dim
        self.img_dim = img_dim
        self.tanh = nn.Tanh()
        flat_out = reduce(mul, img_dim)
        
        self.net = nn.Sequential(
            KANLinear(noise_dim, 256,
                      grid_size=grid_size,
                      spline_order=spline_order,
                      scale_noise=scale_noise,
                      scale_base=scale_base,
                      scale_spline=scale_spline),
            
            KANLinear(256, 512,
                      grid_size=grid_size,
                      spline_order=spline_order,
                      scale_noise=scale_noise,
                      scale_base=scale_base,
                      scale_spline=scale_spline),
            
            KANLinear(512, flat_out,
                      grid_size=grid_size,
                      spline_order=spline_order,
                      scale_noise=scale_noise,
                      scale_base=scale_base,
                      scale_spline=scale_spline),
        )

    def forward(self, x, update_grid=False):
        for layer in self.net:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return self.tanh(x)


class KAN_Discriminator(nn.Module, FlatImageForwardReshape):
    def __init__(
        self,
        img_dim,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
    ):
        super().__init__()
        flat_in = reduce(mul, img_dim)

        self.sigmoid = nn.Sigmoid()
        self.net = nn.Sequential(
            KANLinear(
                flat_in, 512,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline
            ),

            KANLinear(
                512, 256,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline
            ),

            KANLinear(
                256, 1,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline
            ),
        )

    def forward(self, x, update_grid=False):
        for layer in self.net:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return self.sigmoid(x)
