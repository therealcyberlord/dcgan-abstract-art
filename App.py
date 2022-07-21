import streamlit as st
import torch
import DCGAN
import SRGAN
from Utils import color_histogram_mapping, denormalize_images
import torch.nn as nn
import random

device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")

latent_size = 100
display_width = 512
checkpoint_path = "Checkpoints/150epochs.chkpt"

st.title("Generating Abstract Art")
st.text("start generarting by configuring the settings")

st.sidebar.subheader("Configurations")
seed = st.sidebar.slider('Seed', -1000, 1000, 0)

num_images = st.sidebar.slider('Number of Images', 1, 20, 10)

use_srgan = st.sidebar.selectbox(
    'Apply image enhancement',
    ('Yes', 'No')
)

generate = st.sidebar.button("Generate")

if generate:
    torch.manual_seed(seed)
    random.seed(seed)
    
    sampled_noise = torch.randn(num_images, latent_size, 1, 1, device=device)
    generator = DCGAN.Generator(latent_size).to(device)

    dcgan_checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(dcgan_checkpoint['generator_state_dict'])

    generator.eval()

    with torch.no_grad():
        fakes = generator(sampled_noise).detach()

    # use srgan for super resolution
    if use_srgan == "Yes":
        # restore to the checkpoint
        st.write("Using DCGAN then ESRGAN upscale...")
        esrgan_generator = SRGAN.GeneratorRRDB(channels=3, filters=64, num_res_blocks=23).to(device)
        esrgan_checkpoint = torch.load("Checkpoints/esrgan.pt", map_location=device)
        esrgan_generator.load_state_dict(esrgan_checkpoint)

        esrgan_generator.eval()
        with torch.no_grad():
            enhanced_fakes = esrgan_generator(fakes).detach().cpu()
        color_match = color_histogram_mapping(enhanced_fakes, fakes.cpu())

        for i in range(len(color_match)):
            # denormalize and permute to correct color channel
            st.image(denormalize_images(color_match[i]).permute(1, 2, 0).numpy(), width=display_width)


    # default setting -> vanilla dcgan generation
    if use_srgan == "No":
        fakes = fakes.cpu()
        st.write("Using DCGAN Model...")
        for i in range(len(fakes)):
            st.image(denormalize_images(fakes[i]).permute(1, 2, 0).numpy(), width=display_width)




