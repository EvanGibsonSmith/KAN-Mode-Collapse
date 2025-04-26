import torch
from torchvision.models import inception_v3
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, IterableDataset
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
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
from gan_generator.architectures.kan_models import KAN_Generator
from gan_generator.architectures.kan_mlp_hybrid_models import KAN_MLP_Generator
from gan_generator.architectures.cnn_kan_models import Tiny_ConvCIFAR10_KAN_Generator, Strong_ConvCIFAR10_KAN_Generator, Lightweight_ConvCIFAR10_KAN_Generator

def compute_fid(real_loader, generator, device="cuda"):
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # Real images
    for batch, label in tqdm(real_loader, desc="Real"):
        batch = batch.to(device)
        batch = (batch * 255).clamp(0, 255).to(torch.uint8)
        fid.update(batch, real=True)

    # Fake images
    num_images = len(real_dataset)
    batch_size = 64
    for _ in tqdm(range(num_images // batch_size), desc="Fake"):
        z = torch.randn(batch_size, generator.noise_dim).cuda()
        fake_images = generator(z).detach().cpu()

        # Resize and normalize just like real images
        fake_images = torch.nn.functional.interpolate(fake_images, size=(299, 299), mode="bilinear")
        fake_images_uint8 = (fake_images.clamp(0, 1) * 255).to(torch.uint8)
        fid.update(fake_images_uint8.cuda(), real=False)


    return fid.compute().item()


def compute_fid_over_classes(
    real_loader,
    generator,  # expects a function like: def generator_fn(class_idx): yield batches of images
    class_names=None,
    class_to_idx=None,
    device="cuda"
):
    """
    real_loader: DataLoader with (image, label) tuples
    generator_fn: function that takes a class_idx and yields batches of generated images
    class_names: list of class names to evaluate (e.g. ["deer", "horse"])
    class_to_idx: optional dict mapping class names to integer indices (default is CIFAR10 order)

    Returns: dict[class_name] = FID score
    """

    if class_to_idx is None:
        class_to_idx = {
            'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
        }

    results = {}

    for class_name in class_names:
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
        num_images = len(real_dataset)
        batch_size = 64
        for _ in tqdm(range(num_images // batch_size), desc="Fake"):
            z = torch.randn(batch_size, generator.noise_dim).cuda()
            fake_images = generator(z).detach().cpu()

            # Resize and normalize just like real images
            fake_images = torch.nn.functional.interpolate(fake_images, size=(299, 299), mode="bilinear")
            fake_images_uint8 = (fake_images.clamp(0, 1) * 255).to(torch.uint8)
            fid.update(fake_images_uint8.cuda(), real=False)


        results[class_name] = fid.compute().item()

    return results


if __name__=="__main__":
    # Transform to fit Inception's input expectations (299x299, normalized)
    transform = Compose([
        Resize((299, 299)),
        ToTensor(),
        Normalize([0.5]*3, [0.5]*3)  # Scale [-1, 1] to roughly match GAN output
    ])
    
    batch_size = 64

    # CIFAR10 real images
    real_dataset = CIFAR10(root="./data", train=False, transform=transform, download=True)
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)

    # Get Generator
    folder_path = "./gan_generator/outputs/strong_conv_mlp_fc_cifar_output"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_epoch = 100

    with open(f"{folder_path}/config.yaml", "r") as file:
        gen_config = yaml.load(file, Loader=yaml.FullLoader)["Generator"]["params"]

    generator = Strong_ConvCIFAR10_Generator(**gen_config).to(device)
    generator.load_state_dict(torch.load(f"{folder_path}/models/generators/generator_epoch_{model_epoch}.pth", map_location=device))
    generator.eval()

    fid_score = compute_fid(real_loader, generator)
    print("Overall FID Score: ", fid_score)

    overall_fid = compute_fid(real_loader, generator, device)

    fid_scores = compute_fid_over_classes(
        real_loader,
        generator,
        class_names=[
                    "airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"
                    ]
    )

    for cls, fid in fid_scores.items():
        print(f"{cls}: FID = {fid:.2f}")
