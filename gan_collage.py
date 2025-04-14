import torch
from gan import KANGenerator, MLPGenerator, KANDiscriminator, MLPDiscriminator
from fvcore.nn import FlopCountAnalysis
from flop_hooks import kan_group_op_handle  # replace with actual file

import matplotlib.pyplot as plt

# --- Generate MNIST-like Data from GAN ---
def generate_mnist_data_from_gan():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NOISE_DIM = 100  # Latent space dimension for the GAN
    IMG_DIM = 28 * 28  # Flatten MNIST images (28x28 pixels)
    BATCH_SIZE = 36  # How many images to generate for the collage
    
    # --- Initialize Generator ---
    gen = MLPGenerator(noise_dim=NOISE_DIM, img_dim=IMG_DIM).to(DEVICE)
    
    # If you have a pre-trained model, load it here
    gen.load_state_dict(torch.load('models/gans/d_mlp_g_mlp/generator_mlp.pth', map_location=DEVICE))
    
    # --- Generate noise and create fake images ---
    noise = torch.randn(BATCH_SIZE, NOISE_DIM).to(DEVICE)  # Generate random noise
    fake_images = gen(noise)  # Pass noise through the generator to create images

    # Reshape the generated images to 28x28 for visualization
    fake_images = fake_images.view(-1, 28, 28).cpu().detach().numpy()

    # --- Plot the generated images in a 4x4 grid ---
    fig, axes = plt.subplots(6, 6, figsize=(8, 8))
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

def print_flops():
    noise_dim = 100
    img_dim = 784  # for example, 28x28 images flattened

    # Dummy inputs
    z = torch.randn(1, noise_dim)
    img = torch.randn(1, img_dim)

    # Instantiate models
    kan_gen = KANGenerator(noise_dim, img_dim)
    mlp_gen = MLPGenerator(noise_dim, img_dim)
    kan_disc = KANDiscriminator(img_dim)
    mlp_disc = MLPDiscriminator(img_dim)

    # Run FLOP analysis (MACs = FLOPs / 2 for linear layers)
    kan_gen_flops = FlopCountAnalysis(kan_gen, z)
    mlp_gen_flops = FlopCountAnalysis(mlp_gen, z)
    kan_disc_flops = FlopCountAnalysis(kan_disc, img)
    mlp_disc_flops = FlopCountAnalysis(mlp_disc, img)

    # Print results (MACs = FLOPs / 2)

    # Run FLOP analysis (MACs = FLOPs / 2 for linear layers)
    kan_gen_flops = FlopCountAnalysis(kan_gen, z)
    kan_gen_flops.set_op_handle("KAT_Group", kan_group_op_handle)

    mlp_gen_flops = FlopCountAnalysis(mlp_gen, z)

    kan_disc_flops = FlopCountAnalysis(kan_disc, img)
    kan_gen_flops.set_op_handle("KAT_Group", kan_group_op_handle)
    
    mlp_disc_flops = FlopCountAnalysis(mlp_disc, img)
    
    # Print parameter counts
    print(f"KAN Generator Params: {count_params(kan_gen):,}")
    print(f"MLP Generator Params: {count_params(mlp_gen):,}")
    print(f"KAN Discriminator Params: {count_params(kan_disc):,}")
    print(f"MLP Discriminator Params: {count_params(mlp_disc):,}")

    # Print flop counts NOTE: Not working yet due to triton issues
    #print(f"KAN Generator MACs: {kan_gen_flops.total()}")
    #print(f"MLP Generator MACs: {mlp_gen_flops.total()}")
    #print(f"KAN Discriminator MACs: {kan_disc_flops.total()}")
    #print(f"MLP Discriminator MACs: {mlp_disc_flops.total()}")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("Starting Script")
    # MACs
    #print_flops()
    generate_mnist_data_from_gan()