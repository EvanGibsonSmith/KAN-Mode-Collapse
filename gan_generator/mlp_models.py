import torch
import torch.nn as nn
import torch.nn.functional as F
from helper_classes import ForwardReshape
from functools import reduce
from operator import mul

# --- MLP Generator ---
class MLPGenerator(nn.Module, ForwardReshape):
    def __init__(self, noise_dim, img_dim):
        super().__init__()
        self.noise_dim = noise_dim
        self.img_dim = img_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, reduce(mul, img_dim)), # Flattened output
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# --- MLP Discriminator ---
class MLPDiscriminator(nn.Module, ForwardReshape):
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
    
