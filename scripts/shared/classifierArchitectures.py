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

def getFullyConnected1(encoder, outputShape):
    # a deep larger fully connected layer than the simple model

    encoderOutput = encoder.layers[-1].output
    x = Flatten()(encoderOutput)
    x = Dense(50, activation=tf.nn.relu, name='fc_1')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(20, activation=tf.nn.relu, name='fc_2')(x)
    x = Dropout(rate=0.5)(x)
    outputModel = Dense(outputShape, activation=tf.nn.sigmoid, name='output_layer')(x)
    return outputModel