import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from kat import KAN  # Your custom model
import pandas as pd
import argparse

def get_dataset(name, flatten=False):
    if name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)) if flatten else transforms.Identity()
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        input_size = 28 * 28
        num_classes = 10

    elif name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)) if flatten else transforms.Identity()
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        input_size = 32 * 32 * 3
        num_classes = 10

    else:
        raise ValueError("Unsupported dataset. Choose from ['MNIST', 'CIFAR10'].")

    return train_dataset, test_dataset, input_size, num_classes

if __name__=="__main__":
    print("Cuda: ", torch.cuda.is_available())

    # --- Hyperparameters ---
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATASET = "CIFAR10"  

    flatten = True 
    train_dataset, test_dataset, input_size, num_classes = get_dataset(DATASET, flatten=flatten)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    parameters, accuracies = [], []
    for num_hidden_features in [16, 32, 64, 128, 256, 512]:
        print(f"\nHidden Features: {num_hidden_features}")
        model = KAN(in_features=input_size, hidden_features=num_hidden_features, out_features=num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # --- Training ---
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

        # --- Evaluation ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Final Accuracy: {accuracy:.4f}")

        parameters.append(sum(p.numel() for p in model.parameters()))
        accuracies.append(accuracy)

    df = pd.DataFrame({
        'Parameters': parameters,
        'Accuracies': accuracies
    })
    df.to_csv(f'accuracy_to_parameters_{DATASET.lower()}.csv', index=False)

    print("\nAll Parameters:", parameters)
    print("All Accuracies:", accuracies)
