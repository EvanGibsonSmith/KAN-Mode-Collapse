import torch
import torch.nn as nn
from efficient_kan.conv_kan import KAN_Convolutional_Layer
from .helper_classes import IdentityForwardReshape, HParams

class Strong_ConvCIFAR10_KAN_Generator(nn.Module, IdentityForwardReshape, HParams):
    def __init__(self, noise_dim=100, img_channels=3):
        super().__init__()
        self.noise_dim = noise_dim
        self.img_channels = img_channels

        self.net = nn.Sequential(
            nn.Linear(noise_dim, 512 * 2 * 2),
            nn.ReLU(True),
            nn.Unflatten(1, (512, 2, 2)),

            # Had to make a bit weaker due to memory constraints (although KAN layers do tend to have more parameters)
            nn.Upsample(scale_factor=2, mode='nearest'),
            KAN_Convolutional_Layer(512, 256, kernel_size=(3, 3), padding=(1, 1)),

            nn.Upsample(scale_factor=2, mode='nearest'),
            KAN_Convolutional_Layer(256, 128, kernel_size=(3, 3), padding=(1, 1)),

            nn.Upsample(scale_factor=2, mode='nearest'),
            KAN_Convolutional_Layer(128, 64, kernel_size=(3, 3), padding=(1, 1)),

            nn.Upsample(scale_factor=2, mode='nearest'),
            KAN_Convolutional_Layer(64, img_channels, kernel_size=(3, 3), padding=(1, 1)),

            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)
    

class Strong_ConvCIFAR10_KAN_Discriminator(nn.Module, IdentityForwardReshape, HParams):
    def __init__(self, img_channels=3):
        super().__init__()
        self.disc = nn.Sequential(
            KAN_Convolutional_Layer(img_channels, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            KAN_Convolutional_Layer(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            KAN_Convolutional_Layer(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            KAN_Convolutional_Layer(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1),
            nn.Sigmoid()  # or nn.Identity() if using WGAN loss
        )

    def forward(self, x):
        return self.disc(x)
    
class Lightweight_ConvCIFAR10_KAN_Generator(nn.Module, IdentityForwardReshape, HParams):

    def __init__(self, noise_dim=100, img_channels=3):
        super(Lightweight_ConvCIFAR10_KAN_Generator, self).__init__()
        self.noise_dim = noise_dim
        self.img_channels = img_channels

        self.net = nn.Sequential(
            nn.Linear(noise_dim, 512 * 2 * 2),
            nn.ReLU(True),
            nn.Unflatten(1, (512, 2, 2)),

            nn.Upsample(scale_factor=2, mode='nearest'),

            # Lighter KAN layer
            KAN_Convolutional_Layer(
                in_channels=512,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                grid_size=3,  # Reduced grid size for lighter memory usage
                spline_order=2,  # Lower spline order to reduce complexity
                scale_noise=0.05,  # Reduce noise scale to lower computation
                scale_base=0.8,  # Lower base scale to keep operations efficient
                scale_spline=0.8,  # Scale down spline coefficients to save memory
                base_activation=torch.nn.ReLU,  # Simplified activation
                grid_eps=0.02,
                grid_range=[-1, 1],
                device="cuda"
            ),

            nn.Upsample(scale_factor=2, mode='nearest'),
            KAN_Convolutional_Layer(
                in_channels=256,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                grid_size=3,  # Reduced grid size for lighter memory usage
                spline_order=2,  # Lower spline order to reduce complexity
                scale_noise=0.05,  # Reduce noise scale to lower computation
                scale_base=0.8,  # Lower base scale to keep operations efficient
                scale_spline=0.8,  # Scale down spline coefficients to save memory
                base_activation=torch.nn.ReLU,  # Simplified activation
                grid_eps=0.02,
                grid_range=[-1, 1],
                device="cuda"
            ),

            nn.Upsample(scale_factor=2, mode='nearest'),
                        KAN_Convolutional_Layer(
                in_channels=128,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                grid_size=3,  # Reduced grid size for lighter memory usage
                spline_order=2,  # Lower spline order to reduce complexity
                scale_noise=0.05,  # Reduce noise scale to lower computation
                scale_base=0.8,  # Lower base scale to keep operations efficient
                scale_spline=0.8,  # Scale down spline coefficients to save memory
                base_activation=torch.nn.ReLU,  # Simplified activation
                grid_eps=0.02,
                grid_range=[-1, 1],
                device="cuda"
            ),

            nn.Upsample(scale_factor=2, mode='nearest'),
            KAN_Convolutional_Layer(
                in_channels=64,
                out_channels=img_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                grid_size=3,  # Reduced grid size for lighter memory usage
                spline_order=2,  # Lower spline order to reduce complexity
                scale_noise=0.05,  # Reduce noise scale to lower computation
                scale_base=0.8,  # Lower base scale to keep operations efficient
                scale_spline=0.8,  # Scale down spline coefficients to save memory
                base_activation=torch.nn.ReLU,  # Simplified activation
                grid_eps=0.02,
                grid_range=[-1, 1],
                device="cuda"
            ),

            nn.Tanh()
        )


    def forward(self, x):
        return self.net(x)
    

class LightWeight_ConvCIFAR10_KAN_Discriminator(nn.Module, IdentityForwardReshape, HParams):

    def __init__(self, img_channels=3):
        super().__init__()

        # Define the discriminator network with lighter layers
        self.disc = nn.Sequential(
            KAN_Convolutional_Layer(
                in_channels=img_channels,
                out_channels=32,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                dilation=(1, 1),
                grid_size=3,  # Reduced grid size for lighter memory usage
                spline_order=2,  # Lower spline order to reduce complexity
                scale_noise=0.05,  # Reduce noise scale to lower computation
                scale_base=0.8,  # Lower base scale to keep operations efficient
                scale_spline=0.8,  # Scale down spline coefficients to save memory
                base_activation=torch.nn.ReLU,  # Simplified activation
                grid_eps=0.02,
                grid_range=[-1, 1],
                device="cuda"
            ),
            
            KAN_Convolutional_Layer(
                in_channels=32,
                out_channels=64,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                dilation=(1, 1),
                grid_size=3,  # Reduced grid size for lighter memory usage
                spline_order=2,  # Lower spline order to reduce complexity
                scale_noise=0.05,  # Reduce noise scale to lower computation
                scale_base=0.8,  # Lower base scale to keep operations efficient
                scale_spline=0.8,  # Scale down spline coefficients to save memory
                base_activation=torch.nn.ReLU,  # Simplified activation
                grid_eps=0.02,
                grid_range=[-1, 1],
                device="cuda"
            ),
            nn.BatchNorm2d(64),

            KAN_Convolutional_Layer(
                in_channels=64,
                out_channels=128,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                dilation=(1, 1),
                grid_size=3,  # Reduced grid size for lighter memory usage
                spline_order=2,  # Lower spline order to reduce complexity
                scale_noise=0.05,  # Reduce noise scale to lower computation
                scale_base=0.8,  # Lower base scale to keep operations efficient
                scale_spline=0.8,  # Scale down spline coefficients to save memory
                base_activation=torch.nn.ReLU,  # Simplified activation
                grid_eps=0.02,
                grid_range=[-1, 1],
                device="cuda"
            ),
            nn.BatchNorm2d(128),

            # Remove the last convolutional layer to reduce parameters
            KAN_Convolutional_Layer(
                in_channels=128,
                out_channels=128,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                dilation=(1, 1),
                grid_size=3,  # Reduced grid size for lighter memory usage
                spline_order=2,  # Lower spline order to reduce complexity
                scale_noise=0.05,  # Reduce noise scale to lower computation
                scale_base=0.8,  # Lower base scale to keep operations efficient
                scale_spline=0.8,  # Scale down spline coefficients to save memory
                base_activation=torch.nn.ReLU,  # Simplified activation
                grid_eps=0.02,
                grid_range=[-1, 1],
                device="cuda"
            ),            
            nn.BatchNorm2d(128),

            # Replace Flatten + Linear with AdaptiveAvgPool
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1), # TODO replace with KAN linear?
            nn.Sigmoid()  # or nn.Identity() if using WGAN loss
        )

    def forward(self, x):
        return self.disc(x)


class Tiny_ConvCIFAR10_KAN_Generator(nn.Module, IdentityForwardReshape, HParams):

    def __init__(self, noise_dim=100, img_channels=3):
        super(Tiny_ConvCIFAR10_KAN_Generator, self).__init__()
        self.noise_dim = noise_dim
        self.img_channels = img_channels

        self.net = nn.Sequential(
            nn.Linear(noise_dim, 4 * 2 * 2),
            nn.ReLU(True),
            nn.Unflatten(1, (4, 2, 2)),

            nn.Upsample(scale_factor=2, mode='nearest'),

            # Lighter KAN layer
            KAN_Convolutional_Layer(
                in_channels=4,
                out_channels=4,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                grid_size=3,  # Reduced grid size for lighter memory usage
                spline_order=2,  # Lower spline order to reduce complexity
                scale_noise=0.05,  # Reduce noise scale to lower computation
                scale_base=0.8,  # Lower base scale to keep operations efficient
                scale_spline=0.8,  # Scale down spline coefficients to save memory
                base_activation=torch.nn.ReLU,  # Simplified activation
                grid_eps=0.02,
                grid_range=[-1, 1],
                device="cuda"
            ),

            nn.Upsample(scale_factor=2, mode='nearest'),
            KAN_Convolutional_Layer(
                in_channels=4,
                out_channels=4,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                grid_size=3,  # Reduced grid size for lighter memory usage
                spline_order=2,  # Lower spline order to reduce complexity
                scale_noise=0.05,  # Reduce noise scale to lower computation
                scale_base=0.8,  # Lower base scale to keep operations efficient
                scale_spline=0.8,  # Scale down spline coefficients to save memory
                base_activation=torch.nn.ReLU,  # Simplified activation
                grid_eps=0.02,
                grid_range=[-1, 1],
                device="cuda"
            ),

            nn.Upsample(scale_factor=2, mode='nearest'),
                        KAN_Convolutional_Layer(
                in_channels=4,
                out_channels=4,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                grid_size=3,  # Reduced grid size for lighter memory usage
                spline_order=2,  # Lower spline order to reduce complexity
                scale_noise=0.05,  # Reduce noise scale to lower computation
                scale_base=0.8,  # Lower base scale to keep operations efficient
                scale_spline=0.8,  # Scale down spline coefficients to save memory
                base_activation=torch.nn.ReLU,  # Simplified activation
                grid_eps=0.02,
                grid_range=[-1, 1],
                device="cuda"
            ),

            nn.Upsample(scale_factor=2, mode='nearest'),
            KAN_Convolutional_Layer(
                in_channels=4,
                out_channels=img_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                grid_size=3,  # Reduced grid size for lighter memory usage
                spline_order=2,  # Lower spline order to reduce complexity
                scale_noise=0.05,  # Reduce noise scale to lower computation
                scale_base=0.8,  # Lower base scale to keep operations efficient
                scale_spline=0.8,  # Scale down spline coefficients to save memory
                base_activation=torch.nn.ReLU,  # Simplified activation
                grid_eps=0.02,
                grid_range=[-1, 1],
                device="cuda"
            ),
            nn.Tanh()
        )


    def forward(self, x):
        return self.net(x)    

class Tiny_ConvCIFAR10_KAN_Discriminator(nn.Module, IdentityForwardReshape, HParams):

    def __init__(self, img_channels=3):
        super().__init__()

        # Define the discriminator network with lighter layers
        self.disc = nn.Sequential(
            KAN_Convolutional_Layer(
                in_channels=img_channels,
                out_channels=8,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                dilation=(1, 1),
                grid_size=3,  # Reduced grid size for lighter memory usage
                spline_order=2,  # Lower spline order to reduce complexity
                scale_noise=0.05,  # Reduce noise scale to lower computation
                scale_base=0.8,  # Lower base scale to keep operations efficient
                scale_spline=0.8,  # Scale down spline coefficients to save memory
                base_activation=torch.nn.ReLU,  # Simplified activation
                grid_eps=0.02,
                grid_range=[-1, 1],
                device="cuda"
            ),

            # Replace Flatten + Linear with AdaptiveAvgPool
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # or nn.Identity() if using WGAN loss
        )

    def forward(self, x):
        return self.disc(x)