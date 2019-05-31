from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

def miniUnet(inputShape):
    '''
    a small convolution architecture which has a connection
    between the output of the first conv layer to the input
    of the last convolutional layer
    '''
    inputLayer = Input(shape=inputShape)
    convLayer1_d = Conv2D(16,(3,3),activation='relu',padding='same')(inputLayer)
    poolLayer1_d = MaxPooling2D((2,2),padding='same')(convLayer1_d)
    convLayer2_d = Conv2D(16,(3,3),activation='relu',padding='same')(poolLayer1_d)
    upsample1_a = UpSampling2D((2,2))(convLayer2_d) 
    concated1_a = Concatenate()([upsample1_a,convLayer1_d])
    outputLayer = Conv2D(1,(3,3),activation='sigmoid', padding='same')(concated1_a)

    model = Model(inputLayer, outputLayer)
    model.summary()
    return model