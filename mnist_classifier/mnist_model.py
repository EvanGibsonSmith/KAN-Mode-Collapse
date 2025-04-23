import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Model definition
class MNIST_Classifier(nn.Module):
    def __init__(self):
        super(MNIST_Classifier, self).__init__()
        # Two convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)  # → 32×26×26
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1) # → 64×24×24
        self.pool  = nn.MaxPool2d(2, 2)                         # → 64×12×12
        # Two fully connected layers
        self.fc1   = nn.Linear(64 * 12 * 12, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 2. Training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss   = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} '
                  f'[{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')
    # (logging every 100 batches) :contentReference[oaicite:6]{index=6}

# 3. Test/evaluation function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct   = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy   = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    # (computes sum reduction then averages) :contentReference[oaicite:7]{index=7}

# 4. Main script entrypoint
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training Script')
    parser.add_argument('--batch-size', type=int, default=64, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='testing batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--save-model', action='store_true', default=False, help='save the trained model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if use_cuda else 'cpu')
    # (sets seed and selects device) :contentReference[oaicite:8]{index=8}

    # Data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])  # (standard MNIST normalization) :contentReference[oaicite:9]{index=9}

    train_loader = DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        datasets.MNIST('../data', train=False, transform=transform),
        batch_size=args.test_batch_size, shuffle=False
    )
    # (uses torchvision.datasets and DataLoader) :contentReference[oaicite:10]{index=10}

    # Model, optimizer
    model     = MNIST_Classifier().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # (SGD optimizer from torch.optim) :contentReference[oaicite:11]{index=11}

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # Optionally save the trained model
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
        print("Saved model to mnist_cnn.pt")

    # (model saving via state_dict) :contentReference[oaicite:12]{index=12}
