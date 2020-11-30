import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model, Sequential
from keras.datasets import mnist
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation, Dense, Dropout, Input

def build_generator():
    #initializing the neural network
    generator= Sequential()
    #adding an input layer to the network
    generator.add(Dense(units=256, input_dim=100))
    #activating the layer with LeakyReLU activation function
    generator.add(LeakyReLU(0.2))
    #applying batch Normalization
    generator.add(Dense(units=512))
    #adding the third layer
    generator.add(Dense(units=1024))
    generator.add(LeakyReLU(0.2))
    #the output layer with 784(28x28) nodes
    generator.add(Dense(units=784, activation='tanh'))
    #compiling the generator network with loss and optimizer functions
    generator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.adam(lr=0.0002, beta_1=0.5))
    return generator
