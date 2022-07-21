# Generating Abstract Art Live Demo

The goal of this project is to train a DCGAN to generate new works of abstract art. The implementation will be done in PyTorch. 

This project will utilze the model proposed by the 2015 paper "UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS", which advocates the idea of using strided convolution to replace pooling layers.

Dataset used: https://www.kaggle.com/datasets/bryanb/abstract-art-gallery

# Running the project locally 

Download the libraries used with

``` pip install -r requirements.txt ```


Navigate to the cloned/downloaded directory (DCGAN-Abstract-Art) and run
``` python Main.py [num_images] [--seed=somenumber] [--checkpoint=somenumber] [--srgan]```

eg: ``` python Main.py 20 --seed 90 --checkpoint 150 --srgan``` 

***
the arg values will be in type integer and num_images must be in the range (0, 120], only the argument num_images is required.

checkpoint_num: default: 150, you can also use your own trained checkpoints, see the ipynb notebook to continue training. 

Run ```pip Main.py -h``` for more information

<img src="https://github.com/therealcyberlord/DCGAN/blob/master/GIFS/gan-visulization.gif" width="100%">

# Upscaled with ESRGAN
<img src="https://github.com/therealcyberlord/DCGAN-Abstract-Art/blob/master/esrgan_upscale.png" width="100%">

Sources:
* DCGAN Arxiv Paper: https://arxiv.org/pdf/1511.06434v2.pdf
* ESRGAN Arvix Paper: https://arxiv.org/pdf/1809.00219.pdf
* This also take a lot of inspiration from the PyTorch DCGAN Tutorial, check it out <a href="https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html">here</a>

