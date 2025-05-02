import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from torchvision.models import Inception_V3_Weights
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm

# For loading models
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from gan_generator.architectures.mlp_models import MLPGenerator, StrongMLPGenerator
from gan_generator.architectures.kat_models import GRKANGenerator
from gan_generator.architectures.cnn_models import DCGAN_Generator, Strong_ConvCIFAR10_Generator, Strong_ConvCIFAR10_Generator_GR_KAN_Activations
from gan_generator.architectures.cnn_models import Strong_ConvMNIST_Generator_GR_KAN_Activations
from gan_generator.architectures.cnn_models import Strong_ConvMNIST_Generator
from gan_generator.architectures.kan_models import KAN_Generator
from gan_generator.architectures.kan_mlp_hybrid_models import KAN_MLP_Generator
from gan_generator.architectures.cnn_kan_models import Tiny_ConvCIFAR10_KAN_Generator, Strong_ConvCIFAR10_KAN_Generator, Lightweight_ConvCIFAR10_KAN_Generator
import yaml

import mnist_classifier.mnist_model as mnist_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameters
num_real = 5000      # number of real images
num_fake = 5000      # number of generated images
noise_dim = 100     # adjust to match your generator input
img_shape = (1, 28, 28)

# Load CIFAR-10 real images
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

real_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=num_real, shuffle=True)

# Sample real images
real_images, real_labels = next(iter(real_loader))
real_images, real_labels = real_images.to(device), real_labels.to(device)

# Generate fake images
z = torch.randn(num_fake, noise_dim, device=device)

# Load generator model
folder_path = "./gan_generator/outputs/strong_conv_grkan_activations_mnist_output"
if os.path.exists(os.path.join(folder_path, "PCA_feature_embeddings.png")) or os.path.exists(os.path.join(folder_path, "t-SNE_feature_embeddings.png")):
    print("Path already exists.")
    exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_epoch = 100

with open(f"{folder_path}/config.yaml", "r") as file:
    gen_config = yaml.load(file, Loader=yaml.FullLoader)["Generator"]["params"]

generator = Strong_ConvMNIST_Generator_GR_KAN_Activations(**gen_config).to(device)
generator.load_state_dict(torch.load(f"{folder_path}/models/generators/generator_epoch_{model_epoch}.pth", map_location=device))
generator.eval()

with torch.no_grad():
    fake_images = generator(z)
fake_images = fake_images.to(device)

# Make sure images are in the right range for Inception
def preprocess_for_inception(imgs):
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1, 3, 1, 1)  # Repeat the channel dimension to make it RGB
    imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
    return imgs

# Load inception model
inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT)

# Remove the fully connected (fc) layer to get features
inception_model = inception_model.to(device)
inception_model.fc = nn.Identity()
inception_model.eval() 

real_images_resized = preprocess_for_inception(real_images)

# Resize fake images for Inception if MLP
if fake_images.ndim == 2:  
    fake_images = fake_images.view(-1, img_shape[0], img_shape[1], img_shape[2])  # Reshape to (B, C, H, W)

fake_images_resized = preprocess_for_inception(fake_images)

# Extract Inception features
def get_inception_features(imgs, inception_model, batch_size=64):
    num_images = imgs.shape[0]
    all_features = []

    # Iterate through the images in smaller batches
    for start_idx in tqdm(range(0, num_images, batch_size)):
        end_idx = min(start_idx + batch_size, num_images)
        batch = imgs[start_idx:end_idx]

        batch = batch.to(device)  # Move batch to the correct device

        # Run through InceptionV3 to get features (without the final classification layer)
        with torch.no_grad():
            features = inception_model(batch)
        
        all_features.append(features.cpu())  # Store the features on the CPU to avoid GPU memory overflow

    # Concatenate all features from each batch
    all_features = torch.cat(all_features, dim=0)
    
    return all_features

real_features = get_inception_features(real_images_resized, inception_model)
fake_features = get_inception_features(fake_images_resized, inception_model)

real_features = real_features.cpu().numpy()
fake_features = fake_features.cpu().numpy()

# Get predicted labels for fake images
cifar10_classifier_model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet32', pretrained=True).to(device)
cifar10_classifier_model.eval()

mnist_classifier_model = mnist_model.MNIST_Classifier().to(device)
mnist_classifier_model.load_state_dict(torch.load("mnist_classifier/mnist_cnn.pt", map_location=device))
mnist_classifier_model.eval()

classifier_model = mnist_classifier_model # Pick classifier model here based on dataset
with torch.no_grad():
    preds = classifier_model(fake_images)
fake_labels = preds.argmax(dim=1)

fake_labels = fake_labels.cpu().numpy()
real_labels = real_labels.cpu().numpy()
real_labels = real_labels.astype(int)
fake_labels = fake_labels.astype(int)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
all_features = np.concatenate([real_features, fake_features], axis=0)
all_embeddings = tsne.fit_transform(all_features)

real_embeddings = all_embeddings[:len(real_features)]
fake_embeddings = all_embeddings[len(real_features):]

# Plot t-SNE dimensionailty reduction
print("Plotting t-SNE embeddings...")
plt.figure(figsize=(8, 6))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

scatter1 = ax1.scatter(real_embeddings[:, 0], real_embeddings[:, 1], c=real_labels, cmap='tab10', alpha=0.6)
ax1.set_title('Real CIFAR Images (True Labels)')
legend1 = ax1.legend(*scatter1.legend_elements(), title="Classes", loc="best")
ax1.add_artist(legend1)

scatter2 = ax2.scatter(fake_embeddings[:, 0], fake_embeddings[:, 1], c=fake_labels, cmap='tab10', alpha=0.6)
ax2.set_title('Generated Images (Predicted Labels)')
legend2 = ax2.legend(*scatter2.legend_elements(), title="Classes", loc="best")
ax2.add_artist(legend2)

# Show t-SNE dimensionality reduction
plt.title("t-SNE Projection")
plt.show()
plt.tight_layout()
plt.savefig(os.path.join(folder_path, "t-SNE_feature_embeddings.png"))  # Save the collage

# Apply PCA
print("Plotting PCA embeddings...")
plt.figure(figsize=(8, 6))
pca = PCA(n_components=2)
all_features = np.concatenate([real_features, fake_features], axis=0)
all_embeddings = pca.fit_transform(all_features)

real_embeddings = all_embeddings[:len(real_features)]
fake_embeddings = all_embeddings[len(real_features):]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

scatter1 = ax1.scatter(real_embeddings[:, 0], real_embeddings[:, 1], c=real_labels, cmap='tab10', alpha=0.6)
ax1.set_title('Real CIFAR Images (True Labels)')
legend1 = ax1.legend(*scatter1.legend_elements(), title="Classes", loc="best")
ax1.add_artist(legend1)

scatter2 = ax2.scatter(fake_embeddings[:, 0], fake_embeddings[:, 1], c=fake_labels, cmap='tab10', alpha=0.6)
ax2.set_title('Generated Images (Predicted Labels)')
legend2 = ax2.legend(*scatter2.legend_elements(), title="Classes", loc="best")
ax2.add_artist(legend2)

print(f"Explained variance by PCA: {pca.explained_variance_ratio_}")

# Show PCA dimentionality reduction
plt.title("PCA Projection")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
plt.tight_layout()
plt.savefig(os.path.join(folder_path, "PCA_feature_embeddings.png"))  # Save the collage


plt.show()
