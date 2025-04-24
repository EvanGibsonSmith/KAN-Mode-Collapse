import matplotlib.pyplot as plt
import ast
import os
import torch
import pandas as pd

def plot_training_stats(df, save_folder):
    # Convert the 'Confidence' and 'Predicted Classes' columns from strings to lists
    df['Confidence'] = df['Confidence'].apply(ast.literal_eval)
    df['Predicted Classes'] = df['Predicted Classes'].apply(ast.literal_eval)  # Add this line

    # Plot Confidence 
    plt.figure(figsize=(10, 6))
    df['Avg Confidence'] = df['Confidence'].apply(lambda x: sum(x) / len(x))
    plt.plot(df['Epoch'], df['Avg Confidence'], label='Avg Confidence', color='b', marker='o')
    plt.title('Confidence Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "confidence.png"))
    plt.close()

    # Plot Entropy
    plt.figure(figsize=(10, 6))
    plt.plot(df['Epoch'], df['Entropy'], label='Entropy', color='r', marker='x')
    plt.title('Entropy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "entropy.png"))
    plt.close()

    # Generate histogram of predicted classes across all epochs
    all_classes = sum(df['Predicted Classes'], [])  # Flatten the list of lists

    plt.figure(figsize=(8, 6))
    plt.hist(all_classes, bins=range(11), align='left', rwidth=0.8, edgecolor='black')
    plt.xticks(range(10))
    plt.title('Histogram of Predicted Classes')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "histogram.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df['Epoch'], df['G Loss'], label='Generator Loss', color='green', marker='s')
    plt.plot(df['Epoch'], df['D Loss'], label='Discriminator Loss', color='orange', marker='^')
    plt.title('Generator and Discriminator Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "loss_plot.png"))
    plt.close()

# --- Generate MNIST-like Data from GAN ---
def generate_collage(classifier, 
                    generator,
                    noise_dim,
                    save_folder,
                    device=None,
                    collage_size=4,
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
        img_dim (int): flattened image size (28*28)
        batch_size (int): number of samples to generate
    """
    # Setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = classifier.to(device).eval()

    # Generate fake images
    with torch.no_grad():
        z = torch.randn(batch_size, noise_dim, device=device)
        fake = generator(z)  # shape: [B, 784]
    
    # Reshape & normalize for classifier
    # Assuming classifier expects inputs in [0,1] or standardized
    fake_imgs = fake.view(-1, 1, 28, 28)  # [B,1,28,28]
    # If your classifier was trained on normalized data, apply the same transform:
    # e.g., fake_imgs = (fake_imgs + 1) / 2  # if generator output in [-1,1]
    
    # Classify
    with torch.no_grad():
        logits = classifier(fake_imgs)
        preds = logits.argmax(dim=1).cpu().numpy()  # [B]

    # Plotting
    fig, axes = plt.subplots(collage_size, collage_size, figsize=(8, 8))
    fig.suptitle("GAN Generated MNIST-like Images with Predictions", fontsize=16)
    for i, ax in enumerate(axes.flat):
        # Extract individual image from grid
        # Alternatively, display fake_imgs[i] directly:
        img = fake_imgs[i].cpu().squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Pred: {preds[i]}", fontsize=10)
        ax.axis('off')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(save_folder, "generated_and_classified_collage.png"))  # Save the collage
    plt.show()

if __name__=="__main__":
    save_dir = "gan_generator/outputs/conv_gan_cifar10_output"
    stats_df = pd.read_csv("gan_generator/outputs/conv_gan_cifar10_output/training_stats.csv")
    plot_training_stats(stats_df, save_folder=save_dir)