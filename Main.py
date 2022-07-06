import torch 
from torch.optim import Adam
import torchvision.utils as vutils 
import random 
import Model
import matplotlib.pyplot as plt 
import numpy as np 
import argparse


def main():
    # command line capabilities 

    size_limit = 120
    parser = argparse.ArgumentParser()

    parser.add_argument(f"num_images", help="how many images to generate (max: {size_limit})", type=int)
    parser.add_argument("--seed", help="random seed for generating images", type=int, default=random.randint(0, 1e4))
    parser.add_argument("--checkpoint", help="checkpoint to use {70, 100, 120, 150}", type=int, default=150)

    args = parser.parse_args()

    # adding some constraints 
    assert args.num_images <= size_limit, f"Cannot exceed size limit {size_limit}"
    assert args.num_images > 0, "Num images must be greater than 0"

    # set seed for replicable results s
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    latent_size = 100
    lr = 0.0002
    beta1 = 0.5
    checkpoint_path = f"Checkpoints/{args.checkpoint}epochs.chkpt"


    # check if there is a CUDA-compatible GPU, otherwise use a CPU
    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")

    # sample from the normal distribution 
    sampled_noise  = torch.randn(args.num_images, latent_size, 1, 1, device=device)
    generator = Model.Generator(latent_size).to(device)
    discriminator = Model.Discriminator().to(device)

    # defining the models 
    generator_optim = Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    discriminator_optim = Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    # restore to the checkpoint

    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    generator_optim.load_state_dict(checkpoint['generator_optim_state_dict'])
    discriminator_optim.load_state_dict(checkpoint['discriminator_optim_state_dict'])

    fakes = generator(sampled_noise).detach().cpu()

    # plot generated fakes 

    plt.figure(figsize=(18,18))
    plt.title(f"Seed: {args.seed}")
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(fakes, padding=2, normalize=True),(2,1,0)))
    plt.show()


if __name__ == "__main__":
    main()







