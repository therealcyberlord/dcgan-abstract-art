from xmlrpc.client import Boolean
import torch 
from torch.optim import Adam
import torchvision.utils as vutils 
import random 
import DCGAN
import SRGAN
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
    parser.add_argument("--srgan", help="use super resolution (experimental) 0 for no super resolution, any non-zero value for super resolution", type=int, default=0)

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
    generator = DCGAN.Generator(latent_size).to(device)


    # restore to the checkpoint

    dcgan_checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(dcgan_checkpoint['generator_state_dict'])

    generator.eval()
        
    with torch.no_grad():
        fakes = generator(sampled_noise).detach()

    # use srgan

    if args.srgan != 0:
        srgan_generator = SRGAN.GeneratorResNet().to(device)
        srgan_generator.load_state_dict(torch.load(f"Checkpoints/srgan_10.pth", map_location=device))

        srgan_generator.eval()
        with torch.no_grad():
             enhanced_fakes = srgan_generator(fakes).detach().cpu()

        plt.figure(figsize=(12,12))
        plt.title(f"Seed: {args.seed}")
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(enhanced_fakes, padding=2, normalize=True),(2,1,0)))
        plt.show()
 
    # default setting -> vanilla dcgan generation

    else:
        plt.figure(figsize=(12,12))
        plt.title(f"Seed: {args.seed}")
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(fakes.cpu(), padding=2, normalize=True),(2,1,0)))
        plt.show()


if __name__ == "__main__":
    main()







