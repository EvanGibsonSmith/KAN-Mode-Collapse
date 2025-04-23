import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
from test_gan_models import TestGenerator, TestDiscriminator
from mlp_models import MLPGenerator, MLPDiscriminator
from tqdm import tqdm
import os
from metrics import batch_metrics

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

# --- Training Function ---
def train_gan(Generator,
            Discriminator,
            loader,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            epochs = 50,
            learning_rate = 2e-4,
            noise_dim = 100,
            img_dim = 3 * 32 * 32,
            validation_size = 100,
            save_dir='./training_output'):

    if os.path.exists(save_dir):
        print("Save Directory Already Exists.")
        exit(1)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)  

    # --- Model selection ---
    gen = Generator(noise_dim=noise_dim, img_dim=img_dim).to(device)
    disc = Discriminator(img_dim=img_dim).to(device)

    criterion = nn.BCELoss()
    opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Initialize pandas DataFrame to track stats
    columns = ['Epoch', 'G Loss', 'D Loss', 'Entropy', 'Confidence']
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

            loss_disc = (loss_real + loss_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator
            output = disc(fake).view(-1, 1)
            loss_gen = criterion(output, real_labels)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
        
        print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {loss_disc.item():.4f} | G Loss: {loss_gen.item():.4f}")

        # --- Calculate Classifier Metrics (Entropy and Confidence) ---
        with torch.no_grad():
            # Use a pretrained classifier (e.g., ResNet for CIFAR10)
            # Run the fake images through the classifier
            additional_noise = torch.randn(validation_size, noise_dim).to(device)  # Generate more fake images
            fake_images = gen(additional_noise)

            # Calculate entropy and confidence for the generated images
            fake_images_reshaped = fake_images.view(-1, 1, 28, 28)  # Assuming images are 32x32 and 3 channels (modify as needed)
            confidence_hist_entropy, confidences, predicted_classes = batch_metrics(fake_images_reshaped, device)

            # Save stats to DataFrame
            stats_df.loc[len(stats_df)] = {
                'Epoch': epoch + 1,
                'G Loss': loss_gen.item(),
                'D Loss': loss_disc.item(),
                'Entropy': confidence_hist_entropy.item(),
                'Confidence': [float(x) for x in confidences.numpy()],
                'Predicted Classes': predicted_classes,
            }

        # Save model checkpoints and stats at the end of each epoch
        torch.save(gen.state_dict(), f"{save_dir}/generator_epoch_{epoch + 1}.pth")
        torch.save(disc.state_dict(), f"{save_dir}/discriminator_epoch_{epoch + 1}.pth")

    # Save stats DataFrame to CSV
    stats_df.to_csv(f"{save_dir}/training_stats.csv", index=False)

if __name__=="__main__":
    batch_size = 64

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    cifar_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,)),  
        transforms.Lambda(lambda x: x.view(-1))  
    ])

    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    mnist_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_gan(MLPGenerator, MLPDiscriminator, img_dim=28*28, loader=mnist_loader, save_dir='./gan_generator/gan_output', epochs=50)

    