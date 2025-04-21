import numpy as np
import pandas as pd
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from kat_rational import KAT_Group
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class MLPDiscriminator(nn.Module):
    def __init__(self, img_dim, hidden_dim, num_layers):
        super().__init__()
        self.img_dim = img_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.net = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(num_layers)],
            nn.Linear(hidden_dim, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))

class KANDiscriminator(nn.Module):
    def __init__(self, img_dim, hidden_dim, num_layers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            KAT_Group(mode="identity"),
            nn.Linear(hidden_dim, hidden_dim),
            *[nn.Sequential(KAT_Group(mode="gelu"), nn.Linear(hidden_dim, hidden_dim)) for _ in range(num_layers)], # The KAT_Group may have to have it's groups adjusted for parameter size changes? Not entirely sure how it's calculated but graph should be calibrated regardless
            KAT_Group(mode="gelu"),
            nn.Linear(hidden_dim, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))

# Train the model
def train_model(model, criterion, optimizer, X_train, y_train, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        mse = nn.MSELoss()(outputs, y_test).item()
    return mse

# Main script
def main():
    input_size = 28*28
    output_size = 10
    hidden_sizes = [10, 50, 100, 200, 500]  # Increasing parameter sizes
    layers = [1, 2, 3, 4, 5]  # Number of layers to test
    results_file = "results.csv"

    # Check if results file exists
    if os.path.exists(results_file):
        results = pd.read_csv(results_file)
        expected_columns = ["Model", "Hidden_Size", "Num_Layers", "Total_Params", "BCE"]
        for col in expected_columns:
            if col not in results.columns:
                results[col] = None
                raise ValueError(f"Missing column in results file: {col}")
    else:
        results = pd.DataFrame(columns=["Model", "Hidden_Size", "Num_Layers", "Total_Params", "BCE"])
    # Load dataset
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 2e-4
    IMG_DIM = 28 * 28

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    dataset = datasets.MNIST(root="./data",transform=transform, train=True, download=True)
    # Split dataset into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for hidden_size in hidden_sizes:
        for num_layers in layers:
            print(f"Testing Hidden Size: {hidden_size}, Num Layers: {num_layers}")
            for model_name, ModelClass in [("MLP", MLPDiscriminator), ("KAT", KANDiscriminator)]:
                if not ((results["Model"] == model_name) & (results["Hidden_Size"] == hidden_size) & (results["Num_Layers"] == num_layers)).any():
                    for batch_idx, (X_train, y_train) in enumerate(tqdm(train_loader, desc=f"Training {model_name} | Hidden Size: {hidden_size} | Layers: {num_layers}")):
                        
                        X_train = X_train.view(-1, IMG_DIM).to(DEVICE)
                        y_train = torch.nn.functional.one_hot(y_train, num_classes=10).float().to(DEVICE)
                        model = ModelClass(input_size, hidden_size, num_layers).to(DEVICE)
                        criterion = nn.BCEWithLogitsLoss()
                        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

                        train_model(model, criterion, optimizer, X_train, y_train, epochs=EPOCHS)
                    # Evaluate the model
                    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
                    X_test, y_test = next(iter(test_loader))
                    y_test = torch.nn.functional.one_hot(y_test, num_classes=10).float()
                    bce = evaluate_model(model, X_test.to(DEVICE), y_test.to(DEVICE))
                    # Save the model
                    model_dir = "models"
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(model_dir, f"{model_name}_hidden{hidden_size}_layers{num_layers}.pth")
                    torch.save(model.state_dict(), model_path)
                    # Save results
                    new_row = pd.DataFrame([{
                        "Model": model_name,
                        "Hidden_Size": hidden_size,
                        "Num_Layers": num_layers,
                        "Total_Params": hidden_size * num_layers,
                        "BCE": bce
                    }])
                    results = pd.concat([results, new_row], ignore_index=True)
                    results.to_csv(results_file, index=False)
                    print(f"Model: {model_name}, Hidden Size: {hidden_size}, Num Layers: {num_layers}, BCE: {bce}")
    # Plot results
    plt.figure()
    for model_name in results["Model"].unique():
        model_results = results[results["Model"] == model_name]
        plt.plot(model_results["Total_Params"], model_results["MSE"], label=model_name)
    plt.xlabel("Total Parameters")
    plt.ylabel("Mean Squared Error")
    plt.title("Accuracy vs Parameters")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()