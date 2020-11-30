import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model, Sequential
from keras.datasets import mnist
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation, Dense, Dropout, Input

def build_discriminator():
    #Initializing a neural network
    discriminator=Sequential()

    #Adding an Input layer to the network
    discriminator.add(Dense(units=1024, input_dim=784))

    #Activating the layer with LeakyReLU activation function
    discriminator.add(LeakyReLU(0.2))

    #Adding a dropout layer to reduce overfitting
    discriminator.add(Dropout(0.2))

    #Adding a second layer
    discriminator.add(Dense(units=512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    #Adding a third layer
    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    #Adding a forth layer
    discriminator.add(Dense(units=128))
    discriminator.add(LeakyReLU(0.2))

    #Adding the output layer with sigmoid activation
    discriminator.add(Dense(units=1, activation='sigmoid'))

    #Compiling the Discriminator Network with loss and optimizer functions
    discriminator.compile(loss='binary_crossentropy', optimizer = keras.optimizers.adam(lr=0.0002, beta_1=0.5))

    return discriminator
