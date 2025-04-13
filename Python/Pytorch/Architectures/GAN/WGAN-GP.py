import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.autograd as autograd

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
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 14x14 -> 28x28
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
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),           # 14x14 -> 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),          # 7x7 -> 3x3
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=0)             # 3x3 -> 1x1
            # We don't use Sigmoid in WGAN-GP
        )

    def forward(self, x):
        return self.disc(x).view(-1, 1)

def compute_gradient_penalty(critic, real_samples, fake_samples, device, lambda_gp=10):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    critic_interpolates = critic(interpolates)
    grad_outputs = torch.ones(critic_interpolates.size()).to(device)
    
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 50
batch_size = 256
learning_rate = 0.0001
z_dim = 128
num_examples = 16
n_critic = 5

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

gen_opt = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.0, 0.9))
disc_opt = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.0, 0.9))

batches_done = 0
for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        current_batch_size = real.shape[0]

        noise = torch.randn(current_batch_size, z_dim).to(device)
        fake = gen(noise)
        
        loss_disc = -torch.mean(disc(real)) + torch.mean(disc(fake.detach()))
        gp = compute_gradient_penalty(disc, real.data, fake.data, device, lambda_gp=10)
        loss_disc_total = loss_disc + gp

        disc_opt.zero_grad()
        loss_disc_total.backward()
        disc_opt.step()

        if batches_done % n_critic == 0:
            noise = torch.randn(current_batch_size, z_dim).to(device)
            fake = gen(noise)
            loss_gen = -torch.mean(disc(fake))

            gen_opt.zero_grad()
            loss_gen.backward()
            gen_opt.step()

        batches_done += 1

    print(f"Epoch [{epoch+1}/{epochs}] Loss Critic: {loss_disc.item():.4f} GP: {gp.item():.4f} Loss Gen: {loss_gen.item():.4f}")