import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root=r'\Deep-Learning\Python\Pytorch\datasets', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root=r'\Deep-Learning\Python\Pytorch\datasets', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

def train(epochs, data):
    model.train()
    for epoch in range(1, epochs + 1):
        for x, y in data:
            x, y = x.to(device), y.to(device)
            x = x.view(-1, sequence_length, input_size)

            preds = model(x)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch}/{epochs}] - Loss: {loss.item():.4f}')

def get_accuracy(loader):
    num_corrects = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = x.view(-1, sequence_length, input_size)

            predictions = model(x)
            _, preds = predictions.max(1)
            num_corrects += (y == preds).sum().item()
            num_samples += predictions.size(0)

    print(f'Accuracy: {100 * num_corrects / num_samples:.2f}%')
    model.train()

train(num_epochs, train_loader)
get_accuracy(train_loader)
get_accuracy(test_loader)