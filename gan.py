import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from kat_rational import KAT_Group

# --- KAN Generator ---
class KANGenerator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super().__init__()
        self.net = nn.Sequential(
            KAT_Group(mode="identity"),
            nn.Linear(noise_dim, 256),
            KAT_Group(mode="gelu"),
            nn.Linear(256, 512),
            KAT_Group(mode="gelu"),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# --- MLP Generator ---
class MLPGenerator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# --- MLP Discriminator ---
class MLPDiscriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# --- KAN Discriminator ---
class KANDiscriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.fc1 = nn.Linear(img_dim, 512)
        self.act1 = KAT_Group(mode="identity")
        self.fc2 = nn.Linear(512, 256)
        self.act2 = KAT_Group(mode="gelu")
        self.fc3 = nn.Linear(256, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return self.out_act(x)

# --- Training Function ---
def train_gan(use_kan_gen=True, 
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    batch_size=64,
                    epochs=50,
                    learning_rate=2e-4,
                    noise_dim=100,
                    img_dim=28*28,
                    dataset=None,
                    use_kan_disc=False):

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- Model selection ---
    Generator = KANGenerator if use_kan_gen else MLPGenerator
    Discriminator = KANDiscriminator if use_kan_disc else MLPDiscriminator

    gen = Generator(noise_dim=noise_dim, img_dim=img_dim).to(device)
    disc = Discriminator(img_dim=img_dim).to(device)

    criterion = nn.BCELoss()
    opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # --- Training loop ---
    for epoch in range(epochs):
        for real, _ in loader:
            real = real.to(device)
            batch_size = real.size(0)
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake = gen(noise)

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            disc_real = disc(real).view(-1, 1)
            loss_real = criterion(disc_real, real_labels)

            disc_fake = disc(fake.detach()).view(-1, 1)
            loss_fake = criterion(disc_fake, fake_labels)

            loss_disc = (loss_real + loss_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator
            output = disc(fake).view(-1, 1)
            loss_gen = criterion(output, real_labels)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {loss_disc.item():.4f} | G Loss: {loss_gen.item():.4f}")

    torch.save(gen.state_dict(), f"generator_{'kan' if use_kan_gen else 'mlp'}.pth")
    torch.save(disc.state_dict(), f"discriminator_{'kan' if use_kan_disc else 'mlp'}.pth")

if __name__ == "__main__":
    # Set these to easily switch architecture:

    # For MNIST
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize while still 1×28×28
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784 after
    ])

    #mnist_dataset = datasets.MNIST(root="./data", train=True, transform=mnist_transform, download=True)

    # For CIFAR-10
    cifar10_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # for 3 channels
        transforms.Lambda(lambda x: x.view(-1))  # flatten to 3072
    ])

    cifar10_dataset = datasets.CIFAR10(root="./data", train=True, transform=cifar10_transform, download=True)

    train_gan(dataset=cifar10_dataset, img_dim=3 * 32 * 32, use_kan_gen=True, use_kan_disc=False)  # KAN Generator + MLP Discriminator

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from kat_rational import KAT_Group

# --- KAN Generator ---
class KANGenerator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super().__init__()
        self.net = nn.Sequential(
            KAT_Group(mode="identity"),
            nn.Linear(noise_dim, 256),
            KAT_Group(mode="gelu"),
            nn.Linear(256, 512),
            KAT_Group(mode="gelu"),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# --- MLP Generator ---
class MLPGenerator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# --- MLP Discriminator ---
class MLPDiscriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# --- KAN Discriminator ---
class KANDiscriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.fc1 = nn.Linear(img_dim, 512)
        self.act1 = KAT_Group(mode="identity")
        self.fc2 = nn.Linear(512, 256)
        self.act2 = KAT_Group(mode="gelu")
        self.fc3 = nn.Linear(256, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return self.out_act(x)

# --- Training Function ---
def train_gan(use_kan_gen=True, use_kan_disc=False):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    EPOCHS = 50
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

    # --- Model selection ---
    Generator = KANGenerator if use_kan_gen else MLPGenerator
    Discriminator = KANDiscriminator if use_kan_disc else MLPDiscriminator

    gen = Generator(noise_dim=NOISE_DIM, img_dim=IMG_DIM).to(DEVICE)
    disc = Discriminator(img_dim=IMG_DIM).to(DEVICE)

    criterion = nn.BCELoss()
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # --- Training loop ---
    for epoch in range(EPOCHS):
        for real, _ in loader:
            real = real.to(DEVICE)
            batch_size = real.size(0)
            noise = torch.randn(batch_size, NOISE_DIM).to(DEVICE)
            fake = gen(noise)

            real_labels = torch.ones(batch_size, 1).to(DEVICE)
            fake_labels = torch.zeros(batch_size, 1).to(DEVICE)

            # Train Discriminator
            disc_real = disc(real).view(-1, 1)
            loss_real = criterion(disc_real, real_labels)

            disc_fake = disc(fake.detach()).view(-1, 1)
            loss_fake = criterion(disc_fake, fake_labels)

            loss_disc = (loss_real + loss_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator
            output = disc(fake).view(-1, 1)
            loss_gen = criterion(output, real_labels)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}] | D Loss: {loss_disc.item():.4f} | G Loss: {loss_gen.item():.4f}")

    torch.save(gen.state_dict(), f"generator_{'kan' if use_kan_gen else 'mlp'}.pth")
    torch.save(disc.state_dict(), f"discriminator_{'kan' if use_kan_disc else 'mlp'}.pth")

if __name__ == "__main__":
    # Set these to easily switch architecture:
    train_gan(use_kan_disc=True, use_kan_gen=True)  # KAN Generator + MLP Discriminator
