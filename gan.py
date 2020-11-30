import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model, Sequential
from keras.datasets import mnist
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation, Dense, Dropout, Input


#Stacking The Generator And Discriminator Networks To Form A GAN

def gan_net(generator, discriminator):

    #Setting the trainable parameter of discriminator to False
    discriminator.trainable=False

    #Instantiates a Keras tensor of shape 100 (Noise shape)
    inp = Input(shape=(100,))

    #Feeds the input noise to the generator and stores the output in X
    X = generator(inp)

    #Feeds the output from generator(X) to the discriminator and stores the result in out
    out= discriminator(X)

    #Creates a model include all layers required in the computation of out given inp.
    gan= Model(inputs=inp, outputs=out)

    #Compiling the GAN Network
    gan.compile(loss='binary_crossentropy', optimizer = 'adam')

    return gan
