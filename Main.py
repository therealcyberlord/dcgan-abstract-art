import torch 
from torch.optim import Adam
import torchvision.utils as vutils 
import random 
import Model
import matplotlib.pyplot as plt 
import numpy as np 
from Dataset import DatasetProcessing


if __name__ == "__main__":

    # set seed for replicable results s
    torch.manual_seed(999)
    random.seed(10)
    latent_size = 100
    batch_size = 32
    lr = 0.0002
    width = height = 64
    beta1 = 0.5
    checkpoint_path = "Checkpoints/70epochs.chkpt"
    image_path = "Dataset"

    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")

    # load and preprocess the data 
    dataset_processing = DatasetProcessing(image_path, height, width)
    # print(dataset_processing.__len__())
    data_loader = dataset_processing.create_data_loader(batch_size, shuffle=True)


    sampled_noise  = torch.randn(batch_size, latent_size, 1, 1, device=device)
    generator = Model.Generator(latent_size).to(device)
    discriminator = Model.Discriminator().to(device)

    generator_optim = Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    discriminator_optim = Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    # restore to the checkpoint 
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    generator_optim.load_state_dict(checkpoint['generator_optim_state_dict'])
    discriminator_optim.load_state_dict(checkpoint['discriminator_optim_state_dict'])

    fakes = generator(sampled_noise).detach().cpu()

    # plot generated fakes 
    plt.figure(figsize=(18,18))
    plt.axis("off")
    plt.title("Generated Fakes")
    plt.imshow(np.transpose(vutils.make_grid(fakes, padding=2, normalize=True),(1,2,0)))
    plt.show()






