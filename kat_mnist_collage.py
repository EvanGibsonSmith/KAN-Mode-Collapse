import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from kat_rational import KAT_Group
import torch.nn as nn
from kat import KAN

if __name__=="__main__":
    # --- Load model ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KAN(in_features=784, hidden_features=256, out_features=10).to(DEVICE)
    model.load_state_dict(torch.load("kan_mnist_1.pth", map_location=DEVICE))
    model.eval()

    # --- Load dataset ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)  # use a small batch for the collage

    # --- Grab a batch ---
    images, labels = next(iter(test_loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    # --- Plotting ---
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("Model Predictions vs Ground Truth", fontsize=16)

    for i, ax in enumerate(axes.flat):
        img = images[i].cpu().view(28, 28).numpy()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f"P: {preds[i].item()} / T: {labels[i].item()}", fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("mnist_collage.png")
    plt.show()