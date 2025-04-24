import torch
import torch.nn.functional as F
from torchvision import transforms

# Fixes annoying import issues
import sys; sys.path.insert(0, "/root/projects/kan-mode-collapse")

from mnist_classifier import mnist_model

def batch_metrics(batch: torch.Tensor, classifier_model, device: str = ""):
    """
    Input:
        batch: Tensor of shape (N, 3, 32, 32), unnormalized in [0, 1]
    Returns:
        confidence_hist_entropy: Entropy of the confidence histogram
        confidences: Tensor of shape (N,) - max softmax prob
        predicted_classes: Tensor of shape (N,) - class index
    """
    if device == "":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch = batch.to(device)
    
    # Load pretrained CIFAR-10 ResNet20 model from torch.hub
    #model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    # TODO make more flexible between CIFAR-10 and MNSIT

    # Normalization for CIFAR-10
    cifar10_normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                            std=[0.2023, 0.1994, 0.2010])
    mnist_normalize = transforms.Normalize((0.5,), (0.5,))

    # Apply CIFAR-10 normalization
    for i in range(batch.shape[0]):
        batch[i] = mnist_normalize(batch[i]) # TODO make more flexible for both
    
    with torch.no_grad():
        logits = classifier_model(batch)
        probs = F.softmax(logits, dim=1)

        # Get max softmax probabilities (confidence) and predicted class indices
        # TODO confidences being calculated in a bad way
        confidences, predicted_classes = torch.max(probs, dim=1)

        # --- Calculate Confidence Histogram and Entropy ---
        conf_hist = torch.zeros(10)  # For 10 CIFAR-10 classes
        for cls in predicted_classes:
            conf_hist[cls] += 1

        # Normalize the histogram to create a probability distribution
        conf_dist = conf_hist / conf_hist.sum()

        # Calculate entropy of the confidence histogram
        confidence_hist_entropy = -torch.sum(conf_dist * torch.log(conf_dist + 1e-10))

    return confidence_hist_entropy.cpu(), confidences.cpu(), predicted_classes.cpu()