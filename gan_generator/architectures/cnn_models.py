import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from .helper_classes import IdentityForwardReshape, HParams
from efficient_kan.kan import KANLinear  

class ConvCIFAR10_Discriminator(nn.Module, IdentityForwardReshape, HParams):
    def __init__(self):
        super(ConvCIFAR10_Discriminator, self).__init__()
        # Load pretrained ResNet18
        self.model = resnet18(pretrained=False)
        
        # Modify the first conv layer to handle 32x32 images (kernel=3, stride=1, padding=1)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove the maxpool layer (not helpful for small images) This tip came from ChatGPT.
        self.model.maxpool = nn.Identity()

        # Modify the final fully connected layer for CIFAR-10
        self.model.fc = nn.Linear(512, 1)

    def forward(self, x):
        z = self.model(x)
        return torch.sigmoid(z)

class SmallResNetGenerator(nn.Module, IdentityForwardReshape):
    def __init__(self, noise_dim=100, image_channels=3):
        super(SmallResNetGenerator, self).__init__()
        self.fc = nn.Linear(noise_dim, 256 * 4 * 4)  # Flattened 4x4x256
        self.upsample1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.upsample2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.upsample3 = nn.ConvTranspose2d(64, image_channels, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 4, 4)
        x = self.relu(self.bn1(self.upsample1(x)))
        x = self.relu(self.bn2(self.upsample2(x)))
        x = torch.tanh(self.upsample3(x))
        return x
    
class DCGAN_Generator(nn.Module, IdentityForwardReshape, HParams):
    def __init__(self, noise_dim=100, image_channels=3):
        super(DCGAN_Generator, self).__init__()
        self.model = nn.Sequential(
            # First fully connected layer (latent noise vector -> (batch_size, 256, 4, 4))
            nn.ConvTranspose2d(noise_dim, 256, kernel_size=4, stride=1, padding=0),  # Output size: (256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Second layer (increase spatial size from (256, 4, 4) -> (128, 8, 8))
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Third layer (increase spatial size from (128, 8, 8) -> (64, 16, 16))
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Fourth layer (increase spatial size from (64, 16, 16) -> (3, 32, 32))
            nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output layer: apply Tanh to scale to [-1, 1]
        )

    def forward(self, x):
        # Reshape the input noise vector to match the expected input for ConvTranspose2d
        x = x.view(x.size(0), 100, 1, 1)  # Convert to [batch_size, noise_dim, 1, 1]
        return self.model(x)

    
class DCGAN_Discriminator(nn.Module, IdentityForwardReshape, HParams):
    def __init__(self, image_channels=3):
        super(DCGAN_Discriminator, self).__init__()
        self.model = nn.Sequential(
            # First Conv Layer: (3, 32, 32) -> (64, 16, 16)
            nn.Conv2d(image_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Second Conv Layer: (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Third Conv Layer: (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Fully connected layer: (256, 4, 4) -> Scalar output
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),  # Output a single value
            nn.Sigmoid()  # Apply sigmoid to output a scalar between 0 and 1
        )
    
    def forward(self, x):
        x = self.model(x)
        return x.view(-1)  # Flatten to return a scalar for each image in the batch
    
class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(F.interpolate(x, scale_factor=2))
        x = self.upsample(x)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        return x + identity

class ConvCIFAR10_Generator(nn.Module, IdentityForwardReshape, HParams):
    def __init__(self, noise_dim=100, img_channels=3):
        super(ConvCIFAR10_Generator, self).__init__()
        self.init_size = 4  # Output starts at 4x4
        self.proj = nn.Linear(noise_dim, 256 * self.init_size * self.init_size)

        self.block1 = ResidualBlockUp(256, 128)  # 4 → 8
        self.block2 = ResidualBlockUp(128, 64)   # 8 → 16
        self.block3 = ResidualBlockUp(64, 32)    # 16 → 32

        self.to_img = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, img_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, z):
        x = self.proj(z).view(z.size(0), 256, self.init_size, self.init_size)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.to_img(x)
    

class Strong_ConvCIFAR10_Generator(nn.Module, IdentityForwardReshape, HParams):
    def __init__(self, noise_dim=100, img_channels=3, KAN_fc_layer=False):
        self.noise_dim = noise_dim
        self.img_channels = img_channels

        if KAN_fc_layer:
            linear_layer = nn.Sequential(
                KANLinear(noise_dim, 512 * 2 * 2),  # Replace with your actual KANLinear class
                nn.ReLU(True)
            )
        else:
            linear_layer = nn.Sequential(
                nn.Linear(noise_dim, 512 * 2 * 2),
                nn.ReLU(True)
            )
        
        super().__init__()
        self.gen = nn.Sequential(
            # Input: (N, z_dim) → reshape to (N, 512, 2, 2)
            *linear_layer,
            nn.ReLU(True),
            nn.Unflatten(1, (512, 2, 2)),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (4x4)
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (8x8)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # (16x16)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1),  # (32x32)
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)
    
class Strong_ConvCIFAR10_Discriminator(nn.Module, IdentityForwardReshape, HParams):
    def __init__(self, img_channels=3, KAN_fc_layer=False):
        super().__init__()
        if KAN_fc_layer:
            linear_layer = nn.Sequential(
                KANLinear(512 * 2 * 2, 1)  # Replace with your actual KANLinear class
            )
        else:
            linear_layer = nn.Sequential(
                nn.Linear(512 * 2 * 2, 1)
            )
        
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),  # (16x16)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (8x8)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (4x4)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # (2x2)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            *linear_layer,
            nn.Sigmoid()  # or identity if using WGAN
        )

    def forward(self, x):
        return self.disc(x)

