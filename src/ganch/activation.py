"""
activation.py
~~~~~~~~~~~~~~

This file will core part of the neuron. Inside i will write relu, sigmoid and softmax
functions and their gradients.
"""

# load packages
import numpy as np

class Activation:

    def relu(self, weighted_sum : np.ndarray) -> np.ndarray: # Relu activation function, it will help get only positive numbers, if the value is negative, it will simply put zero
        return np.maximum(weighted_sum, 0)

    def relu_grad(self, activation_function : np.ndarray) -> np.ndarray: # Relu gradient, it will calc gradient of the function, basically if the number if positive it will put 1 otherwise 0
        return (activation_function > 0).astype(activation_function.dtype)

    def sigmoid(self, weighted_sum : np.ndarray) -> np.ndarray: # sigmoid activation function will help me for classification, and the otuput values gonna be between 0 - 1
        return np.exp(-weighted_sum)

    def sigmoid_gradient(self, activation_function : np.ndarray) -> np.ndarray: # the derivative of the sigmoid
        return activation_function * (1 - activation_function)

    def softmax(self, weighted_sum : np.ndarray) -> np.ndarray: # softmax function will help me got classification for cathegorical classifications
        weighted_sum_shifted = weighted_sum - np.max(weighted_sum, axis=1, keepdims=True)
        expZ = np.exp(weighted_sum_shifted)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def softmax_gradient(self, weighted_sum : np.ndarray) -> np.ndarray: # and the gradient fo the softmax
        return weighted_sum[:, :, None] * (np.eye(weighted_sum.shape[1], dtype=weighted_sum.dtype)[None] - weighted_sum[:, None, :])