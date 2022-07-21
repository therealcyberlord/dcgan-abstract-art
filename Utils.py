import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from skimage.exposure import match_histograms
import torch

# contains utility functions that we need in the main program

# matches the color histogram of original and the super resolution output
def color_histogram_mapping(images, references):
    matched_list = []
    for i in range(len(images)):
        matched = match_histograms(images[i].permute(1, 2, 0).numpy(), references[i].permute(1, 2, 0).numpy(),
                                   channel_axis=-1)
        matched_list.append(matched)
    return torch.tensor(np.array(matched_list)).permute(0, 3, 1, 2)


def visualize_generations(seed, images):
    num_images = len(images)
    if num_images <= 16:
        nrow = 2
    else:
        nrow = (num_images - 16) // 16 + 2

    plt.figure(figsize=(16, 16))
    plt.title(f"Seed: {seed}")
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(images, padding=2, nrow=nrow, normalize=True), (2, 1, 0)))
    plt.show()

