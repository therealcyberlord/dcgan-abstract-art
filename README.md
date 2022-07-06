# Generating abstract art using DCGAN

The goal of this project is to train a DCGAN to generate new works of abstract art. The implementation will be done in PyTorch. 

This project will utilze the model proposed by the 2015 paper "UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS", which advocates the idea of using strided convolution to replace pooling layers.

Dataset used: https://www.kaggle.com/datasets/bryanb/abstract-art-gallery

# Running the project locally 

Download the libraries used with

``` pip install -r requirements.txt ```


Navigate to the cloned/downloaded directory (DCGAN-Abstract-Art) and run
``` python Main.py [seed] [num_images] [checkpoint_num]```

eg: ``` python Main.py 1239 32 100 ``` 

***
the arg values will be in type integer and num_images must be in the range [0, 64]

checkpoint_num $\in$ [70, 100, 120, 150]

Run ```pip Main.py -h``` for more information

<img src="https://github.com/therealcyberlord/DCGAN/blob/master/GIFS/gan-visulization.gif" width="100%">

Sources:
* Arxiv Paper: https://arxiv.org/pdf/1511.06434v2.pdf

* This also take a lot of inspiration from the PyTorch DCGAN Tutorial, check it out <a href="https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html">here</a>

