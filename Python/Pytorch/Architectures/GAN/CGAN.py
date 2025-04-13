import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(latent_dim, 256 * 7 * 7),
            nn.ReLU(),
            nn.BatchNorm1d(256 * 7 * 7),
            
            nn.Unflatten(1, (256, 7, 7)),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, img_channels, kernel_size=3, stride=1, padding=1),  # 28x28 -> 28x28
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 7x7 -> 3x3
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=0),  # 3x3 -> 1x1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x).view(-1, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 50
batch_size = 256
learning_rate = 0.0002
num_examples = 16
z_dim = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(
    root=r"\Deep-Learning\Python\Pytorch\datasets",
    train=True,
    transform=transform,
    download=True
)

loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True
)


gen = Generator(latent_dim=z_dim, img_channels=1).to(device)
disc = Discriminator(img_channels=1).to(device)

seed = torch.randn((num_examples, z_dim)).to(device)

gen_opt = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
disc_opt = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))
criterion = nn.BCELoss()

def plot_generated_images(epoch, fixed_noise):
    gen.eval()
    with torch.no_grad():
        fake_images = gen(fixed_noise).cpu().reshape(-1, 1, 28, 28)
    
    fig, axes = plt.subplots(1, num_examples, figsize=(num_examples, 1))
    for i in range(num_examples):
        axes[i].imshow(fake_images[i].squeeze(), cmap="gray")
        axes[i].axis("off")
    plt.savefig(f"generated_epoch_{epoch}.png")
    plt.show()
    gen.train()

def train(data, epochs, alpha, beta):
    for epoch in range(1, epochs+1):
        if epoch == 1: start = time.time()

        for batch_idx, (y_true, _) in enumerate(data):
            y_true = y_true.to(device)

            batch_size = y_true.shape[0]

            noise = torch.randn((batch_size, z_dim)).to(device)
            fake = gen(noise)

            real_preds = disc(y_true)
            fake_preds = disc(fake.detach())

            real_loss = criterion(real_preds, torch.ones_like(real_preds) - alpha)
            fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds) + alpha)
            disc_loss = (real_loss + fake_loss) / 2

            disc_opt.zero_grad()
            disc_loss.backward()
            disc_opt.step()

            fake_preds = disc(fake)
            gen_loss = criterion(fake_preds, torch.ones_like(fake_preds) - beta)

            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()

        if epoch % 5 == 0:
            print(f"Epochs: [{epoch}/{epochs}] | Time: {time.time()-start:.2f} sec | "
                  f"Disc Loss: {disc_loss.item():.4f} | Gen Loss: {gen_loss.item():.4f}")
            plot_generated_images(epoch, seed)
            start = time.time()

train(loader, epochs, 0.1, 0.1)