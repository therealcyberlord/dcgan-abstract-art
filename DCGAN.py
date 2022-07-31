from torch import nn
import torch
import torch.nn.functional as F
    
# Generator Code

ngf = 64
num_channels = 3

class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()

        self.latent_size = latent_size
        self.conv1 = nn.ConvTranspose2d(
            self.latent_size, ngf*8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf*8)
        self.conv2 = nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf*4)
        self.conv3 = nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf*2)
        self.conv4 = nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)

        self.conv5 = nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        return torch.tanh(self.conv5(x))