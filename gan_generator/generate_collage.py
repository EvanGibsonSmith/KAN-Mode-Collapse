import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from mlp_models import MLPGenerator

# --- Generate MNIST-like Data from GAN ---
def generate_mnist_data_from_gan():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NOISE_DIM = 100  # Latent space dimension for the GAN
    IMG_DIM = 28 * 28  # Flatten MNIST images (28x28 pixels)
    BATCH_SIZE = 16  # How many images to generate for the collage
    
    # --- Initialize Generator ---
    gen = MLPGenerator(noise_dim=NOISE_DIM, img_dim=IMG_DIM).to(DEVICE)
    
    # If you have a pre-trained model, load it here
    gen.load_state_dict(torch.load('gan_generator/gan_output_mnist/generator_epoch_50.pth', map_location=DEVICE))
    
    # --- Generate noise and create fake images ---
    noise = torch.randn(BATCH_SIZE, NOISE_DIM).to(DEVICE)  # Generate random noise
    fake_images = gen(noise)  # Pass noise through the generator to create images

    # Reshape the generated images to 28x28 for visualization
    fake_images = fake_images.view(-1, 28, 28).cpu().detach().numpy()

    # --- Plot the generated images in a 4x4 grid ---
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("GAN Generated MNIST-like Images", fontsize=16)

    for i, ax in enumerate(axes.flat):
        img = fake_images[i]
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f"Generated {i+1}", fontsize=10)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("generated_mnist_collage.png")  # Save the collage
    plt.show()

# --- Generate CIFAR-10 like data from GAN --- #
def generate_cifar10_data_from_gan():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NOISE_DIM = 100
    IMG_DIM = 3 * 32 * 32  # Flattened CIFAR-10 images
    BATCH_SIZE = 16

    # --- Initialize Generator ---
    gen = KANGenerator(noise_dim=NOISE_DIM, img_dim=IMG_DIM).to(DEVICE)

    # Load pre-trained weights
    gen.load_state_dict(torch.load('models/CIFAR-10/d_mlp_g_kan/generator_kan.pth', map_location=DEVICE))
    gen.eval()

    # --- Generate fake images ---
    noise = torch.randn(BATCH_SIZE, NOISE_DIM).to(DEVICE)
    fake_images = gen(noise).view(-1, 3, 32, 32).cpu().detach()

    # Denormalize: GAN trained on [-1, 1], so rescale to [0, 1]
    fake_images = (fake_images + 1) / 2

    # --- Plot the images in a grid ---
    grid_img = make_grid(fake_images, nrow=4, padding=2)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_img.permute(1, 2, 0))  # Convert from CHW to HWC
    plt.axis('off')
    plt.title("GAN Generated CIFAR-10-like Images")
    plt.tight_layout()
    plt.savefig("generated_cifar10_collage.png")
    plt.show()

def print_flops():
    noise_dim = 100
    img_dim = 784  # for example, 28x28 images flattened

    # Dummy inputs
    z = torch.randn(1, noise_dim)
    img = torch.randn(1, img_dim)

    # Instantiate models
    mlp_gen = MLPGenerator(noise_dim, img_dim)

if __name__ == "__main__":
    print("Starting Script")
    # MACs
    #print_flops() TODO add parameters to this to make it more flexible
    generate_mnist_data_from_gan()
    #generate_cifar10_data_from_gan()
