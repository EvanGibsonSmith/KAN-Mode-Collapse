import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from mlp_models import MLPGenerator
from kat_models import GRKANGenerator
import os

# Fixes annoying import issues
import sys; sys.path.insert(0, "/root/projects/kan-mode-collapse")

import mnist_classifier.mnist_model as mnist_model

# --- Generate MNIST-like Data from GAN ---
def generate_collage(classifier, 
                    generator,
                    noise_dim,
                    save_folder,
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
    if os.path.exists(os.path.join(save_folder, "generated_and_classified_collage.png")):
        print("Path already exists.")
        return
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = classifier.to(device).eval()

    # Generate fake images
    with torch.no_grad():
        z = torch.randn(batch_size, noise_dim, device=device)
        fake = generator.forward_reshape(z)  # shape: [B, 784]
    
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
            img = (img + 1) / 2# Assumes range (-1, 1) from NN

        ax.imshow(img, cmap='gray')
        ax.set_title(f"Pred: {class_dict[preds[i]]}", fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(save_folder, "generated_and_classified_collage.png"))  # Save the collage
    plt.show()

if __name__ == "__main__":
    print("Starting Script")
    # MACs
    #print_flops() TODO add parameters to this to make it more flexible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get classifier model
    classifier_model = mnist_model.MNIST_Classifier().to(device)
    classifier_model.load_state_dict(torch.load("mnist_classifier/mnist_cnn.pt", map_location=device))
    classifier_model.eval()

    cifar10_classifier_model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet32', pretrained=True).to(device)
    cifar10_classifier_model.eval()

    # Get generator model
    noise_dim = 100
    img_dim = (3, 32, 32)
    generator_model = MLPGenerator(noise_dim, img_dim).to(device)
    generator_model.load_state_dict(torch.load("gan_generator/outputs/mlp_gan_cifar10_output/models/generator/generator_epoch_100.pth", map_location=device))
    generator_model.eval()

    cifar10_class_dict = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
    ]

    mnist_class_dict = [
    0,1,2,3,4,5,6,7,8,9
    ]

    generate_collage(cifar10_classifier_model, 
                     generator_model, 
                     noise_dim=100,
                     class_dict=cifar10_class_dict,
                     save_folder="gan_generator/outputs/mlp_gan_cifar10_output")
    
