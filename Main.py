import torch 
from torch.optim import Adam
import torchvision.utils as vutils 
import random 
import DCGAN
import SRGAN
from Utils import visualize_generations, color_histogram_mapping
import matplotlib.pyplot as plt 
import numpy as np 
import argparse


def main():
    # command line capabilities 

    size_limit = 120
    parser = argparse.ArgumentParser()

    parser.add_argument(f"num_images", help="how many images to generate (max: {size_limit})", type=int)
    parser.add_argument("--seed", help="random seed for generating images", type=int, default=random.randint(0, 1e4))
    parser.add_argument("--checkpoint", help="checkpoint to use", type=int, default=150)
    parser.add_argument("--srgan", help="use super resolution (experimental)", action="store_true")

    args = parser.parse_args()

    # adding some constraints 
    assert args.num_images <= size_limit, f"Cannot exceed size limit {size_limit}"
    assert args.num_images > 0, "Num images must be greater than 0"

    # set seed for replicable results s
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    latent_size = 100
    checkpoint_path = f"Checkpoints/{args.checkpoint}epochs.chkpt"


    # check if there is a CUDA-compatible GPU, otherwise use a CPU
    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")

    # sample from the normal distribution 
    sampled_noise  = torch.randn(args.num_images, latent_size, 1, 1, device=device)
    generator = DCGAN.Generator(latent_size).to(device)


    dcgan_checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(dcgan_checkpoint['generator_state_dict'])

    generator.eval()
        
    with torch.no_grad():
        fakes = generator(sampled_noise).detach()

    # use srgan

    if args.srgan:
         # restore to the checkpoint
        esrgan_generator = SRGAN.GeneratorRRDB(channels=3, filters=64, num_res_blocks=23).to(device)
        esrgan_checkpoint = torch.load("Checkpoints/esrgan.pt", map_location=device)
        esrgan_generator.load_state_dict(esrgan_checkpoint)

        # use the esrgan to perform super resolution

        esrgan_generator.eval()
        with torch.no_grad():
             enhanced_fakes = esrgan_generator(fakes).detach().cpu()
        color_match = color_histogram_mapping(enhanced_fakes, fakes.cpu())
        visualize_generations(args.seed, color_match)

    # default setting -> vanilla dcgan generation
    else:
        visualize_generations(args.seed, fakes.cpu())


if __name__ == "__main__":
    main()


