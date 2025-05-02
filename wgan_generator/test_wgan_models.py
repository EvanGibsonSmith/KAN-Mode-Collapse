import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Simple Generator and Discriminator Definitions ---
class TestGenerator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super(TestGenerator, self).__init__()
        self.fc1 = nn.Linear(noise_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, img_dim)
        
    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Output image between -1 and 1
        return x

class TestDiscriminator(nn.Module):
    def __init__(self, img_dim):
        super(TestDiscriminator, self).__init__()
        self.fc1 = nn.Linear(img_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output between 0 and 1 for real/fake
        return x