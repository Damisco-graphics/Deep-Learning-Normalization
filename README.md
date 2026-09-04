# Deep Learning Normalization

Implementation and experimental comparison of **Batch Normalization, Layer Normalization, and Weight Normalization** using TensorFlow operations and their underlying mathematical formulations.

## Overview

This project explores how different normalization strategies affect the training of neural networks. Instead of relying solely on TensorFlow's built-in normalization layers, the normalization operations are implemented directly from their mathematical formulations.

The implementations are integrated into a neural network training pipeline and evaluated against TensorFlow's corresponding normalization functions.

## Implemented Methods

### Batch Normalization

Batch Normalization normalizes activations across a mini-batch by computing the mean and variance for each feature:

$$
\hat{x}_i = \frac{x_i-\mu_{MB}}{\sqrt{\sigma^2_{MB}+\epsilon}}
$$

The normalized activations are then scaled and shifted using learnable parameters:

$$
z_i = \gamma\hat{x}_i+\beta
$$

### Weight Normalization

Weight Normalization reparameterizes a weight vector using a learnable scalar magnitude and direction:

$$
w = \frac{g}{\|v\|}v
$$

This separates the magnitude of the weights from their direction during optimization.

### Layer Normalization

Layer Normalization normalizes across the feature dimension of an individual input rather than across a mini-batch. This provides a different normalization behavior from Batch Normalization and can be useful when batch-dependent statistics are undesirable.

## Experiments

The project evaluates the normalization approaches by:

* Training a convolutional neural network with and without normalization
* Comparing Batch Normalization, Layer Normalization, and Weight Normalization
* Implementing forward passes using basic TensorFlow operations
* Using `tf.GradientTape` for the backward pass
* Comparing custom implementations against TensorFlow's normalization functions
* Comparing gradients between custom and TensorFlow implementations
* Measuring differences between the implementations
* Evaluating the resulting training performance

## Implementation Approach

The custom normalization functions are integrated directly into the model's forward pass rather than treated as isolated demonstrations.

The training pipeline uses TensorFlow's automatic differentiation through `tf.GradientTape`, allowing the custom operations to participate in end-to-end gradient-based optimization.

## Dataset

The experiments use an image classification dataset supported by the project configuration:

* Fashion-MNIST
* CIFAR-10

The selected dataset can be documented here based on the dataset used in the final experiment.

## Technologies

* Python
* TensorFlow 2+
* NumPy
* Convolutional Neural Networks
* Automatic Differentiation
* Gradient Descent
* Batch Normalization
* Layer Normalization
* Weight Normalization

## Key Concepts

This project provides practical experience with:

* Neural network optimization
* Normalization and reparameterization
* Forward and backward propagation
* Automatic differentiation
* CNN training
* Numerical implementation of mathematical formulations
* Experimental comparison of machine learning techniques

## Project Structure

```text
deep-learning-normalization/
│
├── README.md
├── *.py
└── ...
```

The Python implementation contains the custom normalization functions, model definition, training procedure, and experimental comparisons.

## Results

The experiments compare the effect of different normalization strategies on model training and evaluate whether the custom implementations produce results and gradients comparable to TensorFlow's implementations.

Detailed experimental results, plots, and observations can be added here as the project outputs are organized.

## Future Improvements

Potential extensions include:

* Testing additional datasets
* Evaluating normalization methods across different CNN architectures
* Comparing convergence speed and training stability
* Investigating normalization behavior under different batch sizes
* Benchmarking computational overhead
* Extending the implementation to additional normalization techniques.

## References

1. Ioffe, S. & Szegedy, C. *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*, 2015.
2. Ba, J. L., Kiros, J. R. & Hinton, G. E. *Layer Normalization*, 2016.
3. Salimans, T. & Kingma, D. P. *Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks*, 2016.
