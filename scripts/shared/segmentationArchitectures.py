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

def midiUnet(inputShape):
    '''
    a medium sized convolutional architecture with multiple skip 
    connects as in the Unet architecture
    *d* is descending branch of U
    *a* is ascending branch of U
    '''
    inputLayer = Input(shape=inputShape)
    convLayer1_d = Conv2D(8,(5,5),activation='relu',padding='same')(inputLayer)
    poolLayer1_d = MaxPooling2D((2,2),padding='same')(convLayer1_d)
    convLayer2_d = Conv2D(16,(3,3),activation='relu',padding='same')(poolLayer1_d)
    poolLayer2_d = MaxPooling2D((2,2),padding='same')(convLayer2_d)
    convLayer3_d = Conv2D(32,(3,3),activation='relu',padding='same')(poolLayer2_d)
    poolLayer3_d = MaxPooling2D((2,2), padding='same')(convLayer3_d)
    convLayer4_d = Conv2D(32,(3,3),activation='relu',padding='same')(poolLayer3_d)

    upsampLayer1_a = UpSampling2D((2,2))(convLayer4_d)
    stitchLayer1_a = Concatenate()([upsampLayer1_a,convLayer3_d])
    convLayer5_a = Conv2D(16,(3,3), activation='relu',padding='same')(stitchLayer1_a)
    upsampLayer2_a = UpSampling2D((2,2))(convLayer5_a)
    stitchLayer2_a = Concatenate()([upsampLayer2_a,convLayer2_d])
    convLayer6_a = Conv2D(8,(3,3),activation='relu',padding='same')(stitchLayer2_a)
    upsampLayer3_a = UpSampling2D((2,2))(convLayer6_a)
    stitchLayer3_a = Concatenate()([upsampLayer3_a,convLayer1_d])
    outputLayer = Conv2D(1,(3,3),activation='sigmoid',padding='same')(stitchLayer3_a)

    model = Model(inputLayer, outputLayer)
    model.summary()
    return model