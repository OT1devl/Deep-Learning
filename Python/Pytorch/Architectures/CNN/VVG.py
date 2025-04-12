import torch
import torch.nn as nn

VGG_types = {
    'VGG11': (64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'),
    'VGG13': (64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'),
    'VGG16': (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'),
    'VGG19': (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M')
}

class VGG_net(nn.Module):
    def __init__(self, in_channels, num_classes, architecture='VGG16'):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layer(VGG_types[architecture])

        self.fc_input_dim = self.get_fc_input_dim()
        self.fcs = nn.Sequential(
            nn.Linear(self.fc_input_dim, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        return x

    def create_conv_layer(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if isinstance(x, int):
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU()
                ]
                in_channels = x
            else:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def get_fc_input_dim(self):
        with torch.no_grad():
            x = torch.randn(1, self.in_channels, 224, 224)
            x = self.conv_layers(x)
            return x.view(1, -1).shape[1]

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG_net(in_channels=3, num_classes=1000).to(device)
    x = torch.randn(1, 3, 224, 224).to(device)
    print(model(x).shape)