# Perceptron

## Prerequisites
[Python3.5 64 bit](https://www.python.org/downloads)

## Installing

### Pip
If you do not have pip, you can install it by running

```
python3.5 get-pip.py
```

### Python requirements
Once pip is installed you can download the requirements for this project with

```
pip3.5 install -r requirements.txt
```

## Running
The functions to implement for the exercises are located in perceptron_src/Perceptron_td.py. Once you are satisfied with your implementation, you can run the program using:

```
python3.5 main.py
```

## Instructions

### Introduction

In 2012 Krizhevsky et al. achieved groundbreaking results in the field of image classification using a Convolutional Neural Network. The paper *imagenet-classification-with-deep-convolutional-neural-networks* explains how they achieved these results.

### Goal

The goal here is to reimplement a version of this network that will be trained and tested on the CIFAR-10 image classification dataset.

- 1st, using the [dataset documentation](http://www.cs.toronto.edu/~kriz/cifar.html), you will load the data from the cifar-10-batches-py directory and preprocess it so that it can be used with the Keras framework
- 2nd, using the [layers file documentation](https://code.google.com/archive/p/cuda-convnet/wikis/LayerParams.wiki) you will reimplement the network described by the layers-18pct.cfg and layer-params-18pct.cfg files
- 3rd following the method described in the paper, you will perform data augmentation of the CIFAR-10 dataset to improve your results.
- Bonus, Try ro improve the model to achieve the best possible results on the test batch


