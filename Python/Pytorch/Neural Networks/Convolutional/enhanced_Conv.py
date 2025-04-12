import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Enhanced_CNN(nn.Module):
    def __init__(self, inputs, outputs):
        super(Enhanced_CNN, self).__init__()

        self.conv1 = nn.Conv2d(inputs, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, outputs)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = torch.flatten(x, start_dim=1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10
LR = 0.01
BATCH_SIZE = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root=r'\Deep-Learning\Python\Pytorch\datasets', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root=r'\Deep-Learning\Python\Pytorch\datasets', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = Enhanced_CNN(inputs=3, outputs=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def train(data, epochs):
    for epoch in range(1, epochs + 1):
        for x, y in data:
            x, y = x.to(device), y.to(device)

            predictions = model(x)
            loss = criterion(predictions, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch: [{epoch}/{epochs}]> Loss: {loss.item():.4f}')

def get_accuracy(loader):
    num_corrects = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            predictions = model(x)
            _, preds = predictions.max(1)
            num_corrects += (y == preds).sum().item()
            num_samples += predictions.size(0)

    print(f'Accuracy: {100 * num_corrects / num_samples:.2f}%')
    model.train()

train(train_loader, EPOCHS)
get_accuracy(train_loader)
get_accuracy(test_loader)