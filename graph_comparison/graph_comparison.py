import torch
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kat_rational import KAT_Group
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class KANDiscriminator(nn.Module):
    def __init__(self, img_dim, hidden_dim, num_layers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            KAT_Group(mode="identity"),
            nn.Linear(hidden_dim, hidden_dim),
            *[nn.Sequential(KAT_Group(mode="gelu"), nn.Linear(256, 256)) for _ in range(num_layers)], # The KAT_Group may have to have it's groups adjusted for parameter size changes? Not entirely sure how it's calculated but graph should be calibrated regardless
            KAT_Group(mode="gelu"),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

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
    input_size = 10
    output_size = 1
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
    NOISE_DIM = 100
    IMG_DIM = 28 * 28

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for hidden_size in hidden_sizes:
        for num_layers in layers:
            for model_name, ModelClass in [("MLP", MLPDiscriminator), ("KAT", KANDiscriminator)]:
                if not ((results["Model"] == model_name) & (results["Hidden_Size"] == hidden_size)).any():
                    for data, _ in loader:
                        # Prepare data
                        data, labels = data.to(DEVICE), _.to(DEVICE)
                        X_train, y_train = data, labels.float().view(-1, 1)
                        X_test, y_test = data, labels.float().view(-1, 1)

                        model = ModelClass(input_size, hidden_size, num_layers).to(DEVICE)
                        
                        criterion = nn.BCELoss()
                        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

                        train_model(model, criterion, optimizer, X_train, y_train, epochs=EPOCHS)
                        bce = evaluate_model(model, X_test, y_test)

                        # Save results
                        results = results.append({
                            "Model": model_name,
                            "Hidden_Size": hidden_size,
                            "Num_Layers": num_layers,
                            "Total_Params": hidden_size * num_layers,
                            "BCE": bce
                        }, ignore_index=True)
                        results.to_csv(results_file, index=False)

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