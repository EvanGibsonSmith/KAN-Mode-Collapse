import torch
import torch.nn as nn
import torch.nn.functional as F
from .helper_classes import FlatImageForwardReshape, HParams
from functools import reduce
from operator import mul

# --- MLP Generator ---
class MLPGenerator(nn.Module, FlatImageForwardReshape, HParams):
    def __init__(self, noise_dim, img_dim):
        super().__init__()
        self.noise_dim = noise_dim
        self.img_dim = img_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, reduce(mul, img_dim)),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# --- MLP Discriminator ---
class MLPDiscriminator(nn.Module, FlatImageForwardReshape, HParams):
    def __init__(self, img_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(reduce(mul, img_dim), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    
class StrongMLPGenerator(nn.Module, FlatImageForwardReshape, HParams):
    def __init__(self, noise_dim, img_dim):
        super().__init__()
        self.noise_dim = noise_dim
        self.img_dim = img_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Linear(4096, reduce(mul, img_dim)),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class StrongMLPDiscriminator(nn.Module, FlatImageForwardReshape, HParams):
    def __init__(self, img_dim):
        super().__init__()
        self.img_dim = img_dim
        self.net = nn.Sequential(
            nn.Linear(reduce(mul, img_dim), 2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
