import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from gan_generator.architectures.mlp_models import MLPGenerator, StrongMLPGenerator
from gan_generator.architectures.kat_models import GRKANGenerator
from gan_generator.architectures.cnn_models import DCGAN_Generator, Strong_ConvCIFAR10_Generator
from gan_generator.architectures.cnn_models import Strong_ConvMNIST_Generator
from gan_generator.architectures.kan_models import KAN_Generator
from gan_generator.architectures.kan_mlp_hybrid_models import KAN_MLP_Generator
from gan_generator.architectures.cnn_kan_models import Tiny_ConvCIFAR10_KAN_Generator, Strong_ConvCIFAR10_KAN_Generator, Lightweight_ConvCIFAR10_KAN_Generator
import yaml
import os

# Fixes annoying import issues
import sys; sys.path.insert(0, "/root/projects/kan-mode-collapse")

import mnist_classifier.mnist_model as mnist_model

# --- Generate MNIST-like Data from GAN ---
def generate_collage(classifier, 
                    generator,
                    noise_dim,
                    save_folder,
                    save_name="generated_and_classified_collage.png",
                    device=None,
                    class_dict=[0,1,2,3,4,5,6,7,8,9],
                    batch_size=16):
    """
    Generates MNIST-like images with a GAN generator, classifies them with `classifier`,
    and plots a 4×4 grid of images with predicted labels.
    
    Args:
        classifier (nn.Module): pretrained MNIST classifier
        # TODO add save_path documentation
        gen_checkpoint (str): path to generator .pth
        device (torch.device or str): device for computation
        noise_dim (int): latent dimension
        img_dim (tuple): image size like (28, 28)
        batch_size (int): number of samples to generate
    """
    # Setup
    if os.path.exists(os.path.join(save_folder, save_name)):
        print("Path already exists.")
        return
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = classifier.to(device).eval()

    # Generate fake images
    with torch.no_grad():
        z = torch.randn(batch_size, noise_dim, device=device)
        fake = generator.forward_reshape(z) 
    
    # Reshape & normalize for classifier
    # Assuming classifier expects inputs in [0,1] or standardized
    # If your classifier was trained on normalized data, apply the same transform:
    # e.g., fake_imgs = (fake_imgs + 1) / 2  # if generator output in [-1,1]
    
    # Classify
    with torch.no_grad():
        logits = classifier(fake)
        preds = logits.argmax(dim=1).cpu().numpy()  # [B]

    # Plotting
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("GAN Generated MNIST-like Images with Predictions", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        # Extract individual image from grid
        # Alternatively, display fake_imgs[i] directly:
        img = fake[i].cpu().squeeze().numpy()
        if (len(img)==3): # If RGB image
            img = img.transpose(1, 2, 0)  # Convert from (3, x, x) to (x, x, 3)
            img = (img + 1) / 2 # Assumes range (-1, 1) from NN

        ax.imshow(img, cmap='gray')
        ax.set_title(f"Pred: {class_dict[preds[i]]}", fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(save_folder, save_name))  # Save the collage
    plt.show()

if __name__ == "__main__":
    print("Starting Script")
    # MACs
    #print_flops() TODO add parameters to this to make it more flexible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get classifier model
    mnist_classifier_model = mnist_model.MNIST_Classifier().to(device)
    mnist_classifier_model.load_state_dict(torch.load("mnist_classifier/mnist_cnn.pt", map_location=device))
    mnist_classifier_model.eval()

    cifar10_classifier_model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet32', pretrained=True).to(device)
    cifar10_classifier_model.eval()

    # Get generator model
    folder_path = "gan_generator/outputs/strong_conv_kan_fc_mnist_output_few_epoch"
    model_epoch = 6

    # Load yaml for generator hyperparameters
    with open(f"{folder_path}/config.yaml", "r") as file:
        gen_config = yaml.load(file, Loader=yaml.FullLoader)["Generator"]["params"]

    generator_model = Strong_ConvMNIST_Generator(**gen_config).to(device)
    generator_model.load_state_dict(torch.load(f"{folder_path}/models/generators/generator_epoch_{model_epoch}.pth", map_location=device))
    generator_model.eval()

    cifar10_class_dict = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
    ]

    mnist_class_dict = [
    0,1,2,3,4,5,6,7,8,9
    ]

    generate_collage(mnist_classifier_model, 
                     generator_model, 
                     noise_dim=100,
                     class_dict=mnist_class_dict,
                     save_folder=folder_path,
                     save_name=f"generated_and_classified_collage_epoch_{model_epoch}.png")
    
