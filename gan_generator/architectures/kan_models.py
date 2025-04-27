
import torch
import torch.nn as nn
from functools import reduce
from operator import mul
from .helper_classes import FlatImageForwardReshape, HParams
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

class StrongKANGenerator(nn.Module, FlatImageForwardReshape, HParams):
    def __init__(self, noise_dim, img_dim):
        super().__init__()
        self.noise_dim = noise_dim
        self.img_dim = img_dim
        self.net = nn.Sequential(
            KANLinear(noise_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            KANLinear(1024, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            KANLinear(2048, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            KANLinear(4096, reduce(mul, img_dim)),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class StrongKANDiscriminator(nn.Module, FlatImageForwardReshape, HParams):
    def __init__(self, img_dim):
        super().__init__()
        self.img_dim = img_dim
        self.net = nn.Sequential(
            KANLinear(reduce(mul, img_dim), 2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            KANLinear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            KANLinear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            KANLinear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
