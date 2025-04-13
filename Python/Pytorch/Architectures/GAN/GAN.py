import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        nn.init.constant_(m.bias, 0)

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 3e-4
z_dim = 64
image_dim = 28 * 28
batch_size = 128
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
disc.apply(init_weights)
gen.apply(init_weights)

fixed_noise = torch.randn((batch_size, z_dim)).to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root=r"\Deep-Learning\Python\Pytorch\datasets", transform=transform, download=True)
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, image_dim).to(device)
        batch_size = real.shape[0]

        noise = torch.randn((batch_size, z_dim)).to(device)
        fake = gen(noise)

        disc_real = disc(real).view(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real) * 0.9)

        disc_fake = disc(fake.detach()).view(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        output = disc(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss D: {loss_disc.item():.4f}, Loss G: {loss_gen.item():.4f}")

    if epoch % 5 == 0 or epoch == num_epochs - 1:
        with torch.no_grad():
            fake_images = gen(fixed_noise).view(-1, 1, 28, 28).cpu()
        grid = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0), cmap="gray")
        plt.title(f"Epoch {epoch+1}")
        plt.axis("off")
        plt.show()