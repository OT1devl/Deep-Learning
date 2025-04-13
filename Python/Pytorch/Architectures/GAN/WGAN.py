import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

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
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=0)  # 3x3 -> 1x1
            # We don't use Sigmoid in WGAN
        )

    def forward(self, x):
        return self.disc(x).view(-1, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 50
batch_size = 256
learning_rate = 0.00005
z_dim = 128
num_examples = 16
n_critic = 5
clip_value = 0.01

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

gen_opt = optim.RMSprop(gen.parameters(), lr=learning_rate)
disc_opt = optim.RMSprop(disc.parameters(), lr=learning_rate)

batches_done = 0
for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        current_batch_size = real.shape[0]

        noise = torch.randn(current_batch_size, z_dim).to(device)
        fake = gen(noise)

        # loss_critic = - E[D(real)] + E[D(fake)]
        loss_disc = -torch.mean(disc(real)) + torch.mean(disc(fake.detach()))
        
        disc_opt.zero_grad()
        loss_disc.backward()
        disc_opt.step()

        for p in disc.parameters():
            p.data.clamp_(-clip_value, clip_value)

        if batches_done % n_critic == 0:
            noise = torch.randn(current_batch_size, z_dim).to(device)
            fake = gen(noise)
            # loss_gen = - E[D(G(z))]
            loss_gen = -torch.mean(disc(fake))

            gen_opt.zero_grad()
            loss_gen.backward()
            gen_opt.step()

        batches_done += 1

    print(f"Epoch [{epoch+1}/{epochs}] Loss Critic: {loss_disc.item():.4f} Loss Gen: {loss_gen.item():.4f}")