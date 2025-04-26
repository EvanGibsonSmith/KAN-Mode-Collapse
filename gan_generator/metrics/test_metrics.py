import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from metrics import batch_metrics

# Transform: Just ToTensor (normalization is handled in classify_batch)
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load CIFAR-10 test dataset
cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
batch_size = 512
test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)

# Get a batch of images and labels
images, labels = next(iter(test_loader))

# Run classification
entropies, confidences, predicted_classes = batch_metrics(images)

# Print entropy and confidence for each image in the batch
print("Entropy for each image:")
print(entropies)

print("\nConfidence for each image:")
print(confidences)

# Plot histogram of predicted classes
plt.figure(figsize=(8, 5))
plt.hist(predicted_classes.numpy(), bins=np.arange(11) - 0.5, rwidth=0.8, color='skyblue', edgecolor='black')
plt.xticks(np.arange(10))
plt.xlabel("Predicted Class")
plt.ylabel("Frequency")
plt.title("Histogram of Predicted Classes in CIFAR-10 Batch")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
plt.savefig("class_histogram.png")
plt.close()
