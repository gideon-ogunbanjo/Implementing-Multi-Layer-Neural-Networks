# NN-16-1 Pooling on CIFAR100

This project implements experiment 2 of the NN-16-1 pooling method described in the paper "Multi Layer Neural Networks as Replacement for Pooling Operations" by Wolfgang Fuhl and Enkelejda Kasneci. It applies this novel pooling technique to the CIFAR100 dataset.

## Overview

The NN-16-1 pooling method replaces traditional pooling operations (like max or average pooling) with a small neural network. This network consists of 16 neurons in the first layer and 1 neuron in the output layer, serving as a learnable pooling operation.

## Key Features

- Custom `NN_16_1_Pooling` layer implementing the paper's proposed pooling method
- Convolutional Neural Network (CNN) architecture incorporating the custom pooling layer
- Training loop with SGD optimizer and learning rate scheduling
- CIFAR100 dataset loading and preprocessing

## Implementation Details

- Framework: PyTorch
- Dataset: CIFAR100
- Network Architecture:
  - 3 Convolutional layers
  - 3 NN-16-1 Pooling layers
  - 1 Fully connected layer
- Training:
  - 160 epochs
  - Learning rate reduction at 80 and 120 epochs

## Results

This implementation aims to reproduce the results from the original paper, which showed competitive performance with only 194 additional parameters compared to traditional pooling methods cited.

## Usage

1. Ensure PyTorch and torchvision are installed.
2. Run the script to train the network on CIFAR100.
3. The training progress will be printed every 200 mini-batches.

## Future Work

- Implement other pooling methods from the paper for comparison
- Add evaluation on the test set
- Experiment with different network architectures and hyperparameters

## Article Link
 [Fuhl, W., & Kasneci, E. (2021). Multi Layer Neural Networks as Replacement for Pooling Operations. arXiv preprint arXiv:2006.06969v4.](https://arxiv.org/pdf/2006.06969).


 ## Creator
 [Gideon Ogunbanjo](https://gideonogunbanjo.netlify.app)


