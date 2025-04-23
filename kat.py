from kat_rational import KAT_Group
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class KAN(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_cfg=dict(type="KAT", act_init=["identity", "gelu"]),
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act1 = KAT_Group(mode = act_cfg['act_init'][0])
        self.drop1 = nn.Dropout(drop)
        self.act2 = KAT_Group(mode = act_cfg['act_init'][1])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

        # Initialize linear layers and freeze them
        self.initialize_linear_layers()
        
    def initialize_linear_layers(self):
        """ Initialize the linear layers to identity matrix and freeze their weights """
        # Initialize fc1 (identity matrix)
        nn.init.eye_(self.fc1.weight)  # Initialize fc1 weight to identity matrix
        self.fc1.bias.data.zero_()  # Initialize fc1 bias to zero
        self.fc1.requires_grad_(False)  # Freeze fc1 parameters (make untrainable)

        # Initialize fc2 (identity matrix)
        nn.init.eye_(self.fc2.weight)  # Initialize fc2 weight to identity matrix
        self.fc2.bias.data.zero_()  # Initialize fc2 bias to zero
        self.fc2.requires_grad_(False)  # Freeze fc2 parameters (make untrainable)

    def forward(self, x):
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x
        
class KAN_MLP(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_cfg=dict(type="KAT", act_init=["identity", "gelu"]),
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act1 = KAT_Group(mode = act_cfg['act_init'][0])
        self.drop1 = nn.Dropout(drop)
        self.act2 = KAT_Group(mode = act_cfg['act_init'][1])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x
        
if __name__=="__main__":
    print("Cuda: ", torch.cuda.is_available())

    # Check this
    # --- Hyperparameters ---
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 to 784
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model ---
    model = KAN(in_features=784, hidden_features=256, out_features=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for name, param in model.named_parameters():
        print(f"Layer: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")

    exit(1) #NOTE: Not doing training loop anymore
    # --- Training loop ---
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

    print(f"Percent Correct Final Model: {correct/total}")

    torch.save(model.state_dict(), "kan_mnist.pth")
