import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class DNN(nn.Module):
    def __init__(self, inputs, outputs):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(inputs, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_inputs = 784
n_classes = 10
lr = .001
batch_size = 64
epochs = 20

train_dataset = datasets.MNIST(root=r'\Deep-Learning\Python\Pytorch\datasets',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root=r'\Deep-Learning\Python\Pytorch\datasets',
                               train=False,
                               transform=transforms.ToTensor(),
                               download=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = DNN(inputs=n_inputs, outputs=n_classes).to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

def train(epochs):
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            scores = model(data)
            loss = criterion(scores, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss every 10 batches
            # if batch_idx % 10 == 0:  
                # print(f'Epoch: [{epoch}/{epochs}] Batch: {batch_idx} Loss: {loss.item():.4f}')

        print(f'Epoch: [{epoch}/{epochs}]> Loss: {loss.item():.4f}')

def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on train set')
    else:
        print('Checking accuracy on test set')

    num_corrects = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_corrects += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Obtained {num_corrects} / {num_samples} with an accuracy of {float(num_corrects)/float(num_samples):%}')
        model.train()

train(epochs=epochs)

check_accuracy(loader=train_loader, model=model)
check_accuracy(loader=test_loader, model=model)