import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import torch.autograd as autograd
import torch.nn.functional as F
import yaml

# Models
import sys; sys.path.insert(0, "/root/projects/kan-mode-collapse")

from gan_generator.architectures.mlp_models import MLPGenerator, MLPDiscriminator, StrongMLPGenerator, StrongMLPDiscriminator
from gan_generator.architectures.kat_models import GRKANDiscriminator, GRKANGenerator
from gan_generator.architectures.kan_models import KAN_Discriminator, KAN_Generator, StrongKANGenerator, StrongKANDiscriminator
from gan_generator.architectures.cnn_models import ConvCIFAR10_Generator, ConvCIFAR10_Discriminator, DCGAN_Discriminator, DCGAN_Generator, Strong_ConvCIFAR10_Generator, Strong_ConvCIFAR10_Discriminator
from gan_generator.architectures.cnn_kan_models import Strong_ConvCIFAR10_KAN_Generator, Strong_ConvCIFAR10_KAN_Discriminator
from gan_generator.architectures.cnn_models import Strong_ConvCIFAR10_Generator_GR_KAN_Activations, Strong_ConvCIFAR10_Discriminator_GR_KAN_Activations
from gan_generator.architectures.cnn_models import Strong_ConvMNIST_Generator_GR_KAN_Activations, Strong_ConvMNIST_Discriminator_GR_KAN_Activations
from gan_generator.architectures.cnn_models import Strong_ConvMNIST_Generator, Strong_ConvMNIST_Discriminator
from gan_generator.architectures.cnn_kan_models import Lightweight_ConvCIFAR10_KAN_Generator, LightWeight_ConvCIFAR10_KAN_Discriminator
from gan_generator.architectures.cnn_kan_models import Tiny_ConvCIFAR10_KAN_Generator, Tiny_ConvCIFAR10_KAN_Discriminator
from gan_generator.architectures.kan_mlp_hybrid_models import KAN_MLP_Discriminator, KAN_MLP_Generator
from tqdm import tqdm
import os

from gan_generator.metrics.gan_visualizations import plot_training_stats, generate_collage
from gan_generator.metrics.metrics import batch_metrics

# Fixes annoying import issues
import sys; sys.path.insert(0, "/root/projects/kan-mode-collapse")

import mnist_classifier.mnist_model as mnist_model

# ==================Definition End======================

def calculate_gradient_penalty(netD, real_data, fake_data, lambda_hp=10, device="cuda"):
    # Added support for conv layers
    if len(real_data.size()) == 4:  # Conv case: [batch_size, channels, height, width]
        alpha = torch.rand(real_data.size(0), 1, 1, 1).to(device)  # Shape [batch_size, 1, 1, 1]
        alpha = alpha.expand(-1, real_data.size(1), real_data.size(2), real_data.size(3))  # Expand to [batch_size, channels, height, width]
    elif len(real_data.size()) == 2:  # Fully connected case: [batch_size, features]
        alpha = torch.rand(real_data.size(0), 1).to(device)  # Shape [batch_size, 1]
        alpha = alpha.expand(-1, real_data.size(1))  # Expand to [batch_size, features]
    else:
        raise ValueError(f"Unsupported real_data shape: {real_data.size()}")

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.requires_grad_(True).to(device)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_hp
    return gradient_penalty

def get_infinite_batches(data_loader):
    while True:
        for i, (images, _) in enumerate(data_loader):
            yield images
                
def train_loop(netD, netG, Classifier, save_dir, d_lr, g_lr, batch_size, epochs, train_loader, 
               validation_size, noise_dim):

        optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(0.5, 0.9))
        optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(0.5, 0.9))

        # Now batches are callable self.data.next()
        data = get_infinite_batches(train_loader)

        one = torch.tensor(1, dtype=torch.float).to(device)
        mone = one * -1

        # Initialize pandas DataFrame to track stats
        columns = ['Epoch', 'G Loss', 'D Loss', 'Entropy', 'Confidence', 'Predicted Classes']
        stats_df = pd.DataFrame(columns=columns)
        for epoch in range(epochs):

            for i, images in enumerate(train_loader):  # data_loader should be a PyTorch DataLoader

                # === Discriminator === #                 
                # Requires grad, Generator requires_grad = False
                for p in netD.parameters():
                    p.requires_grad = True
                
                netD.zero_grad()

                images = data.__next__()
                # Check for batch to have full batch_size
                if (images.size()[0] != batch_size):
                    continue

                z = torch.rand((batch_size, noise_dim))

                images, z = images.to(device), z.to(device)

                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real images
                d_loss_real = netD(images)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images
                z = torch.randn(batch_size, noise_dim).to(device)
                fake_images = netG(z)
                d_loss_fake = netD(fake_images)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # Train with gradient penalty
                gradient_penalty = calculate_gradient_penalty(netD, images.data, fake_images.data, lambda_hp)
                gradient_penalty.backward()

                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                optimizerD.step()

                # === Generator === 
                for p in netD.parameters():
                    p.requires_grad = False  # to avoid computation

                netG.zero_grad()
                # train generator
                # compute loss with fake images
                
                z = torch.randn(batch_size, epochs).to(device)
                fake_images = netG(z)
                g_loss = netD(fake_images)
                g_loss = g_loss.mean()
                g_loss.backward(mone)
                g_cost = -g_loss
                optimizerG.step()

            print(f'Epoch: [{epoch+1}/{epochs}], G Loss: {g_loss}, D Loss: {d_loss}, Wass. Dist. {Wasserstein_D.cpu().detach().numpy().item()}')
        
            # --- Calculate Classifier Metrics per Epoch (Entropy and Confidence) ---
            with torch.no_grad():
                # Use a pretrained classifier (e.g., ResNet for CIFAR10)
                # Run the fake images through the classifier
                additional_noise = torch.randn(validation_size, noise_dim).to(device)  # Generate more fake images
                fake_images = netG.forward_reshape(additional_noise)

                # Calculate entropy and confidence for the generated images
                confidence_hist_entropy, confidences, predicted_classes = batch_metrics(fake_images, Classifier, device)

                # Save stats to DataFrame
                stats_df.loc[len(stats_df)] = {
                    'Epoch': epochs + 1,
                    'G Loss': g_loss.cpu().data.numpy().item(),
                    'D Loss': d_loss.cpu().data.numpy().item(),
                    'Wass. Distance': Wasserstein_D.cpu().detach().numpy().item(),
                    'Entropy': confidence_hist_entropy.item(),
                    'Confidence': [float(x) for x in confidences.numpy()],
                    'Predicted Classes': [int(x) for x in predicted_classes.numpy()],
                }

            # Save model checkpoints and stats at the end of each epoch
            torch.save(netG.state_dict(), f"{save_dir}/models/generators/generator_epoch_{epoch + 1}.pth")
            torch.save(netD.state_dict(), f"{save_dir}/models/discriminators/discriminator_epoch_{epoch + 1}.pth")

        # Save stats DataFrame to CSV
        stats_df.to_csv(f"{save_dir}/training_stats.csv", index=False)

# --- Training Function ---
def train_gan(Generator,
            Discriminator,
            Classifier,
            loader,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            epochs = 50,
            g_lr = 1e-4, # From the WGAN paper lr for g and d
            d_lr = 1e-4, 
            noise_dim = 100,
            lambda_hp = 10,
            critic_iters=5,
            validation_size = 100,
            save_dir='./training_output'):

    if os.path.exists(save_dir):
        print("Save Directory Already Exists.")
        exit(1)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)  
    os.makedirs(f"{save_dir}/models")
    os.makedirs(f"{save_dir}/models/generators")
    os.makedirs(f"{save_dir}/models/discriminators")

    # Easier model variable names
    netG = Generator
    netD = Discriminator
    
    # Save hyperparameters (with model parameters)
    config = {
        'Generator': {
            'name': netG.__class__.__name__,
            'params': netG.hparams(),
        },
        'Discriminator': {
            'name': netD.__class__.__name__,
            'params': netD.hparams(),
        },
        'classifier': Classifier.__class__.__name__ if Classifier else None,
        'device': str(device),
        'epochs': epochs,
        'g_lr': g_lr,
        'd_lr': d_lr,
        'lambda_hp': lambda_hp,
        'critic_iters': critic_iters,
        'batch_size': loader.batch_size,
        'noise_dim': noise_dim,
        'validation_size': validation_size,
        'save_dir': save_dir,
    }

    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    
    train_loop(netD, netG, Classifier, save_dir, d_lr, g_lr, batch_size, epochs, loader, validation_size, noise_dim)

if __name__=="__main__":
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        #transforms.Lambda(lambda x: x.view(-1)) # Don't flatten for conv networks TODO add flatten into non conv network setups?
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    cifar_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    cifar10_classifier_model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet32', pretrained=True).to(device)
    cifar10_classifier_model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,)),  
        #transforms.Lambda(lambda x: x.view(-1))  # remove this flattening for conv
    ])

    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    mnist_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    mnist_classifier_model = mnist_model.MNIST_Classifier().to(device)
    mnist_classifier_model.load_state_dict(torch.load("mnist_classifier/mnist_cnn.pt", map_location=device))
    mnist_classifier_model.eval()
    
    # --- RUN TRAINING ---
    noise_dim = 100
    img_dim = (3, 32, 32)
    img_channels = 3
    lambda_hp = 10

    # --- MLP CIFAR-10 to see better FID for implementation ---
    train_gan(Strong_ConvCIFAR10_Generator(noise_dim, img_channels).to(device).train(),
            Strong_ConvCIFAR10_Discriminator(img_channels, wgan=True).to(device).train(), 
            cifar10_classifier_model, 
            noise_dim=noise_dim,
            d_lr=1e-4,
            g_lr=1e-4,
            lambda_hp=lambda_hp,
            loader=cifar_loader, save_dir='./gan_generator/outputs/WGAN Fixed/strong_conv_high_lr_mlp_activations_cifar10_output', epochs=100)
    