
import torch
import torch.nn as nn
import torch.nn.functional as F
from helper_classes import ForwardReshape
from functools import reduce
from operator import mul

# Fixes annoying import issues
import sys; sys.path.insert(0, "/root/projects/kan-mode-collapse")
from kat_rational import KAT_Group


# --- KAN Generator ---
class GRKANGenerator(nn.Module, ForwardReshape):
    def __init__(self, noise_dim, img_dim):
        super().__init__()
        self.net = nn.Sequential(
            KAT_Group(mode="identity"),
            nn.Linear(noise_dim, 256),
            KAT_Group(mode="gelu"),
            nn.Linear(256, 512),
            KAT_Group(mode="gelu"),
            nn.Linear(512, reduce(mul, img_dim)),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# --- KAN Discriminator ---
class GRKANDiscriminator(nn.Module, ForwardReshape):
    def __init__(self, img_dim):
        super().__init__()
        self.fc1 = nn.Linear(reduce(mul, img_dim), 512)
        self.act1 = KAT_Group(mode="identity")
        self.fc2 = nn.Linear(512, 256)
        self.act2 = KAT_Group(mode="gelu")
        self.fc3 = nn.Linear(256, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return self.out_act(x)