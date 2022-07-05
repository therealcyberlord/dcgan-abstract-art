from torch import nn
import torch
import torch.nn.functional as F


class DCGAN():
    def __init__(self, generator, discriminator, generator_optim, discriminator_optim, fixed_loss, device):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optim = generator_optim
        self.discriminator_optim = discriminator_optim
        self.fixed_loss = fixed_loss
        self.device = device 

    def generate(self):
        # code for the generator, which generates the inputs
        generation = self.generator(self.fixed_noise)
        return generation

    def train(self, epochs, dataloader):
        criterion = nn.BCELoss()
        real_label = 1. 
        fake_label =  0.

        for i, data in enumerate(dataloader):
            # train the discriminator 
            self.discriminator.zero_grad()
            real_data = data[0].to(self.device)
            real_data_num = real_data.size(0)
            
            # [1, 1, 1, 1, ... 1]
            label = torch.ones(real_data_num, dtype=torch.float32, device=self.device)
            output = self.discriminator(real_data).view(-1)
            error_real_batch = criterion(output, label)

            error_real_batch.backward()
            D_x = output.mean().item()

            # generating fake data 
            noise = torch.randn(real_data_num, self.generator.latent_size, 1, 1, device=self.device)
            fake_data = self.generator(noise).detach()

            # target label should be 0
            label.fill_(fake_label)
            output = self.discriminator(fake_data).view(-1)
            error_fake_batch = criterion(output, fake_data)
            

            



        


class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()

        self.latent_size = latent_size
        self.conv1 = nn.ConvTranspose2d(
            self.latent_size, 64*8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(64*8)
        self.conv2 = nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64*4)
        self.conv3 = nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64*2)
        self.conv4 = nn.ConvTranspose2d(64*2, 64, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        return torch.tanh(self.conv5(x))

# code for the discriminator, which classifies input as real or fake


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64*2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64*2)
        self.conv3 = nn.Conv2d(64*2, 64*4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64*4)
        self.conv4 = nn.Conv2d(64*4, 64*8, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64*8)
        self.conv5 = nn.Conv2d(64*8, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.bn1(self.conv2(x)),
                         negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv3(x)),
                         negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv4(x)),
                         negative_slope=0.2, inplace=True)
        return torch.sigmoid(self.conv5(x))
