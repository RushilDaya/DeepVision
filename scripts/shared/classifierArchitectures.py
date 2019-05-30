from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from math import ceil
from random import randint

def getSimple(encoder, outputShape):
    # a very simple fully connected part
    # only adds a single output layer to the code (hopes that semantic information is already in the code)

    encoderOutput = encoder.layers[-1].output
    x = Flatten()(encoderOutput)
    outputModel = Dense(outputShape, activation=tf.nn.sigmoid, name='output_layer')(x)
    return outputModel