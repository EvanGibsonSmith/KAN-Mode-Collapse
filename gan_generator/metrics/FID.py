import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale
from torchvision.datasets import CIFAR10, MNIST
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import gc
import yaml

# Models
# Fix to have absolute imports work
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from gan_generator.architectures.mlp_models import MLPGenerator
from gan_generator.architectures.kat_models import GRKANGenerator
from gan_generator.architectures.cnn_models import DCGAN_Generator, Strong_ConvCIFAR10_Generator
from gan_generator.architectures.cnn_models import Strong_ConvMNIST_Generator
from gan_generator.architectures.cnn_models import *
from gan_generator.architectures.kan_models import KAN_Generator
from gan_generator.architectures.kan_mlp_hybrid_models import KAN_MLP_Generator
from gan_generator.architectures.cnn_kan_models import Tiny_ConvCIFAR10_KAN_Generator, Strong_ConvCIFAR10_KAN_Generator, Lightweight_ConvCIFAR10_KAN_Generator

# Fixes annoying import issues
import sys; sys.path.insert(0, "/root/projects/kan-mode-collapse")

import mnist_classifier.mnist_model as mnist_model

def compute_fid(real_loader, generator, device="cuda"):
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # Real images
    for batch, label in tqdm(real_loader, desc="Real"):
        batch = batch.to(device)
        batch = (batch * 255).clamp(0, 255).to(torch.uint8)
        print(batch.shape)
        fid.update(batch, real=True)

    # Fake images
    batch_size = 64
    for _ in tqdm(range(len(real_loader)), desc="Fake"):
        z = torch.randn(batch_size, generator.noise_dim).cuda()
        fake_images = generator(z).detach().cpu()
        if fake_images.shape[1] == 1:
            fake_images = fake_images.repeat(1, 3, 1, 1)

        # Resize and normalize just like real images
        fake_images = torch.nn.functional.interpolate(fake_images, size=(299, 299), mode="bilinear")
        fake_images_uint8 = (fake_images.clamp(0, 1) * 255).to(torch.uint8)
        fid.update(fake_images_uint8.cuda(), real=False)

    fid_score = fid.compute().item()

    # Access the fid mean/covariance from internal state (copied roughly from compute method in FrechetInceptionDistance class)
    mean_real = (fid.real_features_sum / fid.real_features_num_samples).unsqueeze(0)
    mean_fake = (fid.fake_features_sum / fid.fake_features_num_samples).unsqueeze(0)

    cov_real_num = fid.real_features_cov_sum - fid.real_features_num_samples * mean_real.t().mm(mean_real)
    cov_real = cov_real_num / (fid.real_features_num_samples - 1)
    cov_fake_num = fid.fake_features_cov_sum - fid.fake_features_num_samples * mean_fake.t().mm(mean_fake)
    cov_fake = cov_fake_num / (fid.fake_features_num_samples - 1)

    return mean_real, cov_real, mean_fake, cov_fake, fid_score

def compute_inception_score(size_generated, generator, device="cuda"):
    inception_score = InceptionScore().to(device)

    # Fake images
    batch_size = 64
    all_fake_images = []

    for _ in tqdm(range(size_generated), desc="Fake"):
        z = torch.randn(batch_size, generator.noise_dim).cuda()
        fake_images = generator(z).detach().cpu()

        # Resize and normalize just like real images (assuming they are 32x32)
        fake_images = torch.nn.functional.interpolate(fake_images, size=(299, 299), mode="bilinear")
        if fake_images.shape[1] == 1:
            fake_images = fake_images.repeat(1, 3, 1, 1)

        all_fake_images.append(fake_images)

    all_fake_images = torch.cat(all_fake_images, dim=0)

    # Compute Inception Score for generated images
    score = inception_score(all_fake_images)

    return score.item()


def compute_latent_inception_covariances(
    real_loader,    # DataLoader yielding (real_images, real_labels)
    generator,      # Generator model
    classifier,     # Classifier model
    device="cpu",
    batch_size=16,
    num_fake_batches=10,
):
    classifier = classifier.to(device)
    generator = generator.to("cuda") # Must be on CUDA to work with triton for GR-KAN
    classifier.eval()
    generator.eval()
    
    real_images_by_class = {}
    fake_images_by_class = {}

    # --- Step 1: Collect real images grouped by true labels keep in CPU to not overload GPU
    for real_imgs, real_labels in real_loader:
        real_imgs = real_imgs.to("cuda")
        real_labels = real_labels.to("cuda")
        
        for img, label in zip(real_imgs, real_labels):
            label = label.item()
            if label not in real_images_by_class:
                real_images_by_class[label] = []
            real_images_by_class[label].append(img.to("cpu")) # TODO figure out why this can't be sent to CPU

    # --- Step 2: Generate fake images and group by predicted labels ---
    for _ in range(num_fake_batches):
        noise = torch.randn(batch_size, generator.noise_dim, device="cuda") 
        fake_imgs = generator(noise) 

        with torch.no_grad():
            preds = classifier(fake_imgs)
            pred_labels = preds.argmax(dim=1)

        for img, pred in zip(fake_imgs, pred_labels):
            pred = pred.item()
            if pred not in fake_images_by_class:
                fake_images_by_class[pred] = []
            fake_images_by_class[pred].append(img.cpu())

    # --- Step 3: Compute covariance matrices per class ---
    cov_real_by_class = {}
    cov_fake_by_class = {}

    all_classes = set(real_images_by_class.keys()).union(fake_images_by_class.keys())

    for cls in tqdm(all_classes, desc="Classes"):
        fid = FrechetInceptionDistance(feature=2048).to("cuda")
        real_imgs = real_images_by_class.get(cls, [])
        fake_imgs = fake_images_by_class.get(cls, [])

        real_imgs = torch.stack(real_imgs).to("cuda")
        fake_imgs = torch.stack(fake_imgs).to("cuda")

        if real_imgs.shape[0] in (0, 1) or fake_imgs.shape[0] in (0, 1):
            print(f"Warning: 0/1 images for class {cls}. Skipping.")
            continue

        # Add real images for this class in batches
        
        # Resize and normalize just like real images
        fake_images_big = torch.nn.functional.interpolate(fake_imgs, size=(299, 299), mode="bilinear")
        if fake_images_big.shape[1] == 1:
            fake_images_big = fake_images_big.repeat(1, 3, 1, 1)
        fake_images_big_uint8 = (fake_images_big.clamp(0, 1) * 255).to(torch.uint8)

        # Load fake images for this class
        fake_image_class_dataset = TensorDataset(fake_images_big_uint8)
        fake_image_class_loader = DataLoader(fake_image_class_dataset, batch_size=batch_size, shuffle=False)
        for images in tqdm(fake_image_class_loader, desc="Loading Fake Images into InceptionV3"):
            images = images[0] # Images is a list of only image element
            fake_class_batch = images.to(device)
            fake_class_batch = (fake_class_batch * 255).clamp(0, 255).to(torch.uint8)
            fid.update(fake_class_batch, real=False)
        
        # Load real images for this class
        real_image_class_dataset = TensorDataset(real_imgs)
        real_image_class_loader = DataLoader(real_image_class_dataset, batch_size=batch_size, shuffle=False)
        for images in tqdm(real_image_class_loader, desc="Loading Real Images into InceptionV3"):
            images = images[0] # Images is a list of only image element
            real_class_batch = images.to("cuda")
            real_class_batch = (real_class_batch * 255).clamp(0, 255).to(torch.uint8).to("cuda")
            fid.update(real_class_batch, real=True)

        fid_output = fid.compute().item()
        mean_real = (fid.real_features_sum / fid.real_features_num_samples).unsqueeze(0)
        mean_fake = (fid.fake_features_sum / fid.fake_features_num_samples).unsqueeze(0)

        cov_real_num = fid.real_features_cov_sum - fid.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (fid.real_features_num_samples - 1)
        cov_fake_num = fid.fake_features_cov_sum - fid.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (fid.fake_features_num_samples - 1)

        # Compute covariance
        cov_real_by_class[cls] = cov_real
        cov_fake_by_class[cls] = cov_fake
        print("Class: ", cls)
        print("Real Cov: ", torch.norm(cov_real, p='fro'))
        print("Fake Cov: ", torch.norm(cov_fake, p='fro'))

        # --- FREE MEMORY ---
        del real_images_by_class[cls]
        del fake_images_by_class[cls]
        del real_imgs
        del fake_imgs
        torch.cuda.empty_cache()
        gc.collect()

    return cov_real_by_class, cov_fake_by_class


def compute_fid_over_classes(
    real_loader,
    generator,  # expects a function like: def generator_fn(class_idx): yield batches of images
    class_to_idx = {
            'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
        },
    device="cuda"
):
    """
    real_loader: DataLoader with (image, label) tuples
    generator_fn: function that takes a class_idx and yields batches of generated images
    class_names: list of class names to evaluate (e.g. ["deer", "horse"])
    class_to_idx: optional dict mapping class names to integer indices (default is CIFAR10 order)

    Returns: dict[class_name] = FID score
    """
    results = {}

    for class_name in class_to_idx.keys():
        class_idx = class_to_idx[class_name]
        fid = FrechetInceptionDistance(feature=2048).to(device)

        # Real images of just this class
        for images, labels in tqdm(real_loader, desc="Real"):
            mask = labels == class_idx
            if mask.any():
                batch = images[mask].to(device)
                batch = (batch * 255).clamp(0, 255).to(torch.uint8)
                fid.update(batch, real=True)

        # Fake images for this class
        batch_size = 64
        for _ in tqdm(range(len(real_loader)), desc="Fake"):
            z = torch.randn(batch_size, generator.noise_dim).cuda()
            fake_images = generator(z).detach().cpu()

            # Resize and normalize just like real images
            fake_images = torch.nn.functional.interpolate(fake_images, size=(299, 299), mode="bilinear")
            if fake_images.shape[1] == 1:
                fake_images = fake_images.repeat(1, 3, 1, 1)

            fake_images_uint8 = (fake_images.clamp(0, 1) * 255).to(torch.uint8)
            fid.update(fake_images_uint8.cuda(), real=False)

        fid_score = fid.compute().item()
        mean_real = (fid.real_features_sum / fid.real_features_num_samples).unsqueeze(0)
        mean_fake = (fid.fake_features_sum / fid.fake_features_num_samples).unsqueeze(0)

        cov_real_num = fid.real_features_cov_sum - fid.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (fid.real_features_num_samples - 1)
        cov_fake_num = fid.fake_features_cov_sum - fid.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (fid.fake_features_num_samples - 1)
        
        # Store results
        results[class_name] = [mean_real, cov_real, mean_fake, cov_fake, fid_score]

    return results


if __name__=="__main__":
    # Transform to fit Inception's input expectations (299x299, normalized)
    transform = Compose([
        Resize((299, 299)),
        Grayscale(num_output_channels=3), # For MNIST
        ToTensor(),
        Normalize([0.5]*3, [0.5]*3)  # Scale [-1, 1] to roughly match GAN output
    ])
    
    batch_size = 64

    # CIFAR10 real images
    cifar_real_dataset = CIFAR10(root="./data", train=False, transform=transform, download=True)
    cifar_real_loader = DataLoader(cifar_real_dataset, batch_size=batch_size, shuffle=False)

    mnist_real_dataset = MNIST(root="./data", train=False, transform=transform, download=True)
    mnist_real_loader = DataLoader(mnist_real_dataset, batch_size=batch_size, shuffle=False)

    # Get Generator
    folder_path = "./gan_generator/outputs/strong_conv_grkan_activations_mnist_output"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_epoch = 100

    with open(f"{folder_path}/config.yaml", "r") as file:
        gen_config = yaml.load(file, Loader=yaml.FullLoader)["Generator"]["params"]

    generator = Strong_ConvMNIST_Generator_GR_KAN_Activations(**gen_config).to(device)
    generator.load_state_dict(torch.load(f"{folder_path}/models/generators/generator_epoch_{model_epoch}.pth", map_location=device))
    generator.eval()

    #overall_inception_score = compute_inception_score(500, generator)
    #print("Overall Inception Score: ", overall_inception_score)

    #real_mean, real_cov, fake_mean, fake_cov, overall_fid = compute_fid(mnist_real_loader, generator)
    #real_cov_norm = torch.norm(real_cov, p='fro')
    #fake_cov_norm = torch.norm(fake_cov, p='fro')
    #print("Overall FID Score: ", overall_fid)
    #print("Distributions: \n")
    #print("Real Mean: ", real_mean)
    #print("Real Cov: ", real_cov_norm)
    #print("Fake Mean: ", fake_mean)
    #print("Fake Cov: ", fake_cov_norm)
    
    #fid_scores = compute_fid_over_classes(
    #    mnist_real_loader,
    #    generator,
    #    class_to_idx={
    #                "0": 0,
    #                "1": 1, 
    #                "2": 2, 
    #                "3": 3, 
    #                "4": 4, 
    #                "5": 5, 
    #                "6": 6, 
    #                "7": 7,
    #                "8": 8, 
    #                "9": 9,
    #                },
    #)

    # Get classifier model
    mnist_classifier_model = mnist_model.MNIST_Classifier().to(device)
    mnist_classifier_model.load_state_dict(torch.load("mnist_classifier/mnist_cnn.pt", map_location=device))
    mnist_classifier_model.eval()

    cifar10_classifier_model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet32', pretrained=True).to(device)
    cifar10_classifier_model.eval()

    classifier_model = mnist_classifier_model

    cov_real, cov_fake = compute_latent_inception_covariances(
        mnist_real_loader,
        generator,
        classifier_model,
        device="cuda",
        batch_size=batch_size,
        num_fake_batches=1,
    )

    print("Covariances by class: \n")
    for cov_cl in cov_real.keys():
        print("Real Cov: ", cov_real[cov_cl])
        print("Fake Cov: ", cov_fake[cov_cl])

    for cls, [mean_real, cov_real, mean_fake, cov_fake, fid_score] in fid_scores.items():
        print(f"{cls}: FID = {fid_score:.2f}")
        print("Real Cov: ", real_cov_norm)
        print("Fake Cov: ", fake_cov_norm)
