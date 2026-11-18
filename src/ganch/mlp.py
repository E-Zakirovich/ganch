"""
mlp.py
~~~~~~

This folder will collect all other files into one place. Inside of this file, there will be 
1 class. The class will receive the dataset and its output from user.
"""

# load packages and classess
import numpy as np

class MLP:

    def __init__(
            self,
            network_size : list[np.number], # the size of the whole network
            epochs : int, # the number of epochs for training
            ):
        self.network_size = network_size
        self.epochs = epochs

    def __forward(self, X : list):
        pass

    def __backward(self, grad : np.float256):
        pass

    def fit(X : list, Y : list):
        pass