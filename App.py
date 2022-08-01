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
checkpoint_path = "Checkpoints/150epochs.chkpt"

st.title("Generating Abstract Art")

st.sidebar.subheader("Configurations")
seed = st.sidebar.slider('Seed', -100000, 100000, 0)

num_images = st.sidebar.slider('Number of Images', 1, 8, 1)

use_srgan = st.sidebar.selectbox(
    'Apply image enhancement',
    ('Yes', 'No')
)

generate = st.sidebar.button("Generate")
st.write("Get started using the left side bar :sunglasses:")

# caching the expensive model loading 

@st.cache(allow_output_mutation=True)
def load_dcgan():
    model = torch.jit.load('Checkpoints/dcgan.pt', map_location=device)
    return model 

@st.cache(allow_output_mutation=True)
def load_esrgan():
    model_state_dict = torch.load("Checkpoints/esrgan.pt", map_location=device)
    return model_state_dict

# if the user wants to generate something new 
if generate:
    torch.manual_seed(seed)
    random.seed(seed)
    
    sampled_noise = torch.randn(num_images, latent_size, 1, 1, device=device)
    generator = load_dcgan()
    generator.eval()

    with torch.no_grad():
        fakes = generator(sampled_noise).detach()

    # use srgan for super resolution
    if use_srgan == "Yes":
        # restore to the checkpoint
        esrgan_generator = SRGAN.GeneratorRRDB(channels=3, filters=64, num_res_blocks=23).to(device)
        esrgan_checkpoint = load_esrgan()
        esrgan_generator.load_state_dict(esrgan_checkpoint)

        esrgan_generator.eval()
        with torch.no_grad():
            enhanced_fakes = esrgan_generator(fakes).detach().cpu()
        color_match = color_histogram_mapping(enhanced_fakes, fakes.cpu())

        cols = st.columns(num_images)
        for i in range(len(color_match)):
            # denormalize and permute to correct color channel
            cols[i].image(denormalize_images(color_match[i]).permute(1, 2, 0).numpy(), use_column_width=True)


    # default setting -> vanilla dcgan generation
    if use_srgan == "No":
        fakes = fakes.cpu()

        cols = st.columns(num_images)
        for i in range(len(fakes)):
            cols[i].image(denormalize_images(fakes[i]).permute(1, 2, 0).numpy(), use_column_width=True)






