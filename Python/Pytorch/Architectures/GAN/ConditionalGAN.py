import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img+1, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(.2),
            self._block(features_d, features_d*2, 4, 2, 1),
            self._block(features_d*2, features_d*4, 4, 2, 1),
            self._block(features_d*4, features_d*8, 4, 2, 1),
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0)
        )
        self.embed = nn.Embedding(num_classes, img_size*img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(.2)
        )
    
    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, num_classes, img_size, embed_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            self._block(channels_noise+embed_size, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.net(x)
    
def gradient_penalty(critic, labels, real, fake, device='cpu'):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.randn((BATCH_SIZE, 1, 1, 1), device=device).repeat(1, C, H, W)
    interpolated_images = (real * alpha + fake * (1 - alpha)).requires_grad_(True)
    
    mixed_scores = critic(interpolated_images, labels)
    grad_outputs = torch.ones_like(mixed_scores, device=device)
    
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradient = gradient.view(gradient.size(0), -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penal = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penal

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 1e-4
batch_size = 64
img_size = 64
channels_img = 1
num_classes = 10
gen_embedding = 100
z_dim = 100
num_epochs = 100
features_disc = 16
features_gen = 16
critic_iterations = 5
lambda_gp = 10

transforms = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(
    root=r'\Deep-Learning\Python\Pytorch\datasets',
    train=True,
    transform=transforms,
    download=True
)
loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True
)

gen = Generator(z_dim, channels_img, features_gen, num_classes, img_size, gen_embedding).to(device=device)
disc = Discriminator(channels_img, features_disc, num_classes, img_size).to(device=device)

opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.0,0.9))
opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.0,0.9))


def train(num_epochs, loader, gen, disc, opt_gen, opt_disc, device, z_dim, critic_iterations, lambda_gp):
    batches_done = 0
    for epoch in range(num_epochs):
        for batch_idx, (real, real_labels) in enumerate(loader):
            real = real.to(device)
            real_labels = real_labels.to(device)
            current_batch_size = real.shape[0]

            for _ in range(critic_iterations):
                noise = torch.randn(current_batch_size, z_dim, 1, 1, device=device)
                fake = gen(noise, real_labels)
                
                critic_real = disc(real, real_labels)
                critic_fake = disc(fake.detach(), real_labels)
                
                gp = gradient_penalty(disc, real_labels, real, fake.detach(), device=device)
                loss_critic = torch.mean(critic_fake) - torch.mean(critic_real) + lambda_gp * gp

                opt_disc.zero_grad()
                loss_critic.backward()
                opt_disc.step()

            noise = torch.randn(current_batch_size, z_dim, 1, 1, device=device)
            fake = gen(noise, real_labels)
            loss_gen = -torch.mean(disc(fake, real_labels))

            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx}/{len(loader)} "
                      f"Loss Critic: {loss_critic.item():.4f} | Loss Gen: {loss_gen.item():.4f} | GP: {gp.item():.4f}")
            batches_done += 1

train(num_epochs, loader, gen, disc, opt_gen, opt_disc, device, z_dim, critic_iterations, lambda_gp)