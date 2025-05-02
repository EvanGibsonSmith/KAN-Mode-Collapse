import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
import yaml

# Models
from test_gan_models import TestGenerator, TestDiscriminator
from architectures.mlp_models import (
    MLPGenerator,
    MLPDiscriminator,
    StrongMLPGenerator,
    StrongMLPDiscriminator,
)
from architectures.kat_models import GRKANDiscriminator, GRKANGenerator
from architectures.kan_models import (
    KAN_Discriminator,
    KAN_Generator,
    StrongKANGenerator,
    StrongKANDiscriminator,
)
from architectures.cnn_models import (
    ConvCIFAR10_Generator,
    ConvCIFAR10_Discriminator,
    DCGAN_Discriminator,
    DCGAN_Generator,
    Strong_ConvCIFAR10_Generator,
    Strong_ConvCIFAR10_Discriminator,
)
from architectures.cnn_kan_models import (
    Strong_ConvCIFAR10_KAN_Generator,
    Strong_ConvCIFAR10_KAN_Discriminator,
)
from architectures.cnn_models import (
    Strong_ConvCIFAR10_Generator_GR_KAN_Activations,
    Strong_ConvCIFAR10_Discriminator_GR_KAN_Activations,
)
from architectures.cnn_models import (
    Strong_ConvMNIST_Generator_GR_KAN_Activations,
    Strong_ConvMNIST_Discriminator_GR_KAN_Activations,
)
from architectures.cnn_kan_models import (
    Lightweight_ConvCIFAR10_KAN_Generator,
    LightWeight_ConvCIFAR10_KAN_Discriminator,
)
from architectures.cnn_kan_models import (
    Tiny_ConvCIFAR10_KAN_Generator,
    Tiny_ConvCIFAR10_KAN_Discriminator,
)
from architectures.kan_mlp_hybrid_models import KAN_MLP_Discriminator, KAN_MLP_Generator
from tqdm import tqdm
import os

from gan_generator.metrics.gan_visualizations import (
    plot_training_stats,
    generate_collage,
)
from gan_generator.metrics.metrics import batch_metrics

# Fixes annoying import issues
import sys

sys.path.insert(0, "/root/projects/kan-mode-collapse")

import mnist_classifier.mnist_model as mnist_model


# --- Function to Calculate Entropy of Confidence Histogram ---
def entropy_of_confidence_histogram(pred_classes, confidences, num_classes=10):
    # Sum confidence scores per predicted class
    conf_hist = torch.zeros(num_classes)

    for cls, conf in zip(pred_classes, confidences):
        conf_hist[cls] += conf

    # Normalize to get probability distribution
    conf_dist = conf_hist / conf_hist.sum()

    # Compute entropy
    entropy = -torch.sum(conf_dist * torch.log(conf_dist + 1e-10))
    return entropy.item(), conf_dist


def gradient_penalty(D, xr, xf):
    """
    :param D:
    :param xr: [b, 2]
    :param xf: [b, 2]
    :return:
    """
    # [b, 1]
    t = torch.rand(batch_size, 1).cuda()
    # [b, 1] => [b, 2]  broadcasting so t is the same for x1 and x2
    t = t.expand_as(xr)
    # interpolation
    mid = t * xr + (1 - t) * xf
    # set it to require grad info
    mid.requires_grad_()
    pred = D(mid)
    grads = autograd.grad(
        outputs=pred,
        inputs=mid,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()
    return gp


# --- Training Function ---
def train_wgan(
    Generator,
    Discriminator,
    Classifier,
    loader,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    epochs=50,
    g_lr=2e-4,
    d_lr=2e-4,
    noise_dim=100,
    validation_size=100,
    save_dir="./training_output",
):

    if os.path.exists(save_dir):
        print("Save Directory Already Exists.")
        exit(1)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/models")
    os.makedirs(f"{save_dir}/models/generators")
    os.makedirs(f"{save_dir}/models/discriminators")

    # Easier model variable names
    gen = Generator
    disc = Discriminator

    # Save hyperparameters (with model parameters)
    config = {
        "Generator": {
            "name": gen.__class__.__name__,
            "params": gen.hparams(),
        },
        "Discriminator": {
            "name": disc.__class__.__name__,
            "params": disc.hparams(),
        },
        "classifier": Classifier.__class__.__name__ if Classifier else None,
        "device": str(device),
        "epochs": epochs,
        "g_lr": g_lr,
        "d_lr": d_lr,
        "noise_dim": noise_dim,
        "validation_size": validation_size,
        "save_dir": save_dir,
    }

    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    criterion = nn.BCELoss()
    opt_gen = optim.Adam(gen.parameters(), lr=g_lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=d_lr, betas=(0.5, 0.999))

    # Initialize pandas DataFrame to track stats
    columns = [
        "Epoch",
        "G Loss",
        "D Loss",
        "Entropy",
        "Confidence",
        "Predicted Classes",
    ]
    stats_df = pd.DataFrame(columns=columns)

    # --- Training loop ---
    for epoch in tqdm(range(epochs)):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            batch_size = real.size(0)
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake = gen(noise)

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            disc_real = disc(real).view(-1, 1)
            loss_real = criterion(disc_real, real_labels)

            disc_fake = disc(fake.detach()).view(-1, 1)
            loss_fake = criterion(disc_fake, fake_labels)

            gp = gradient_penalty(disc, real, fake)
            # Gradient Penalty for WGAN
            loss_disc = (loss_real + loss_fake) / 2 + 0.2 * gp
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator
            output = disc(fake).view(-1, 1)
            loss_gen = criterion(output, real_labels)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        print(
            f"Epoch [{epoch+1}/{epochs}] | D Loss: {loss_disc.item():.4f} | G Loss: {loss_gen.item():.4f}"
        )

        # --- Calculate Classifier Metrics (Entropy and Confidence) ---
        with torch.no_grad():
            # Use a pretrained classifier (e.g., ResNet for CIFAR10)
            # Run the fake images through the classifier
            additional_noise = torch.randn(validation_size, noise_dim).to(
                device
            )  # Generate more fake images
            fake_images = gen.forward_reshape(additional_noise)

            # Calculate entropy and confidence for the generated images
            confidence_hist_entropy, confidences, predicted_classes = batch_metrics(
                fake_images, Classifier, device
            )

            # Save stats to DataFrame
            stats_df.loc[len(stats_df)] = {
                "Epoch": epoch + 1,
                "G Loss": loss_gen.item(),
                "D Loss": loss_disc.item(),
                "Entropy": confidence_hist_entropy.item(),
                "Confidence": [float(x) for x in confidences.numpy()],
                "Predicted Classes": [int(x) for x in predicted_classes.numpy()],
            }

        # Save model checkpoints and stats at the end of each epoch
        torch.save(
            gen.state_dict(),
            f"{save_dir}/models/generators/generator_epoch_{epoch + 1}.pth",
        )
        torch.save(
            disc.state_dict(),
            f"{save_dir}/models/discriminators/discriminator_epoch_{epoch + 1}.pth",
        )

    # Save stats DataFrame to CSV
    stats_df.to_csv(f"{save_dir}/training_stats.csv", index=False)


if __name__ == "__main__":
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            # transforms.Lambda(lambda x: x.view(-1)) # Don't flatten for conv networks TODO add flatten into non conv network setups?
        ]
    )
    dataset = datasets.CIFAR10(
        root="./data", train=True, transform=transform, download=True
    )
    cifar_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    cifar10_classifier_model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True
    ).to(device)
    cifar10_classifier_model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            # transforms.Lambda(lambda x: x.view(-1))  # remove this flattening for conv
        ]
    )

    dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    mnist_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    mnist_classifier_model = mnist_model.MNIST_Classifier().to(device)
    mnist_classifier_model.load_state_dict(
        torch.load("mnist_classifier/mnist_cnn.pt", map_location=device)
    )
    mnist_classifier_model.eval()

    # --- RUN MNIST ---
    noise_dim = 100
    img_dim = (1, 28, 28)
    img_channels = 1
    train_wgan(
        Strong_ConvMNIST_Generator_GR_KAN_Activations(noise_dim, img_channels)
        .to(device)
        .train(),
        Strong_ConvMNIST_Discriminator_GR_KAN_Activations(img_channels)
        .to(device)
        .train(),
        mnist_classifier_model,
        noise_dim=noise_dim,
        d_lr=2e-5,  # 1/10 of Generator
        loader=mnist_loader,
        save_dir="./gan_generator/outputs/strong_conv_grkan_activations_mnist_output",
        epochs=100,
    )
