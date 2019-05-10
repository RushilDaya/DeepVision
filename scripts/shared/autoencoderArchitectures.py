from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import pickle
import tensorflow as tf
import numpy as np
import os

from shared.autoencoderArchitectures import getNormal, getDayaNet, getLeCunhaNet

def getNormal(inputShape):
    """
    the normal network codes from 200*200*3 to a 25*25*8 code
    :param inputShape: tuple - the shape of the input images
    :return: tuple (model, encodermap, decodermap)
    """
    input_img = Input(shape=inputShape, name='input_layer')
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='enc_conv1')(input_img)
    x = MaxPooling2D((2, 2), padding='same', name='enc_max_pool1')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='enc_conv2')(x)
    x = MaxPooling2D((2, 2), padding='same', name='enc_max_pool2')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='enc_conv3')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name='enc_max_pool3')(x)

    # at this point the representation is (25, 25, 8) - see model summaries

    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='dec_conv1')(encoded)
    x = UpSampling2D((2, 2), name='dec_up_samp1')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='dec_conv2')(x)
    x = UpSampling2D((2, 2), name='dec_up_samp2')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='dec_conv3')(x)
    x = UpSampling2D((2, 2), name='dec_up_samp3')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='output_layer')(x)

    # Generate models
    autoencoder = Model(input_img, decoded)
    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)   
    # create a placeholder for an encoded (32-dimensional) input
    encoding_dim = autoencoder.get_layer('enc_max_pool3').output_shape[1:]

    input_enc = Input(shape=encoding_dim, name='enc_in')
    deco = autoencoder.layers[-7](input_enc)
    deco = autoencoder.layers[-6](deco)
    deco = autoencoder.layers[-5](deco)
    deco = autoencoder.layers[-4](deco)
    deco = autoencoder.layers[-3](deco)
    deco = autoencoder.layers[-2](deco)
    deco = autoencoder.layers[-1](deco)

    # create the decoder model
    decoder = Model(input_enc, deco)

    return (autoencoder, encoder, decoder)


def getDayaNet(inputShape):
    """
    the daya network codes from 200*200*3 to a 5*5*4 code
    :param inputShape: tuple - the shape of the input images
    :return: tuple (model, encodermap, decodermap)
    """
    input_img = Input(shape=inputShape, name='input_layer')
    x = Conv2D(16,(3,3),activation='relu',padding='same', name='enc_conv1')(input_img)
    x = MaxPooling2D((2,2),padding='same', name='enc_max_pool1')(x)
    x = Conv2D(16,(3,3),activation='relu',padding='same', name='enc_conv2')(x)
    x = MaxPooling2D((5,5), padding='same', name='enc_max_pool2')(x)
    x = Conv2D(8,(3,3),activation='relu',padding='same', name='enc_conv3')(x)
    x = Conv2D(4,(3,3),activation='relu',padding='same', name='enc_conv4')(x)
    encoded = MaxPooling2D((4,4),padding='same', name='enc_max_pool3')(x)


    x = UpSampling2D((4,4), name='dec_up_samp1')(encoded)
    x = Conv2D(8,(3,3),activation='relu', padding='same', name='dec_conv1')(x)
    x = Conv2D(16,(3,3),activation='relu', padding='same', name='dec_conv2')(x)
    x = UpSampling2D((5,5),name='dec_up_samp2')(x)
    x = Conv2D(16,(3,3),activation='relu',padding='same',name='dec_conv3')(x)
    x = UpSampling2D((2,2),name='dec_up_samp3')(x)
    decoded = Conv2D(3,(3,3),activation='sigmoid', padding='same', name='dec_conv4')(x)


    # Generate models
    autoencoder = Model(input_img, decoded)
    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)   
    # create a placeholder for an encoded (32-dimensional) input
    encoding_dim = autoencoder.get_layer('enc_max_pool3').output_shape[1:]

    input_dec = Input(shape=encoding_dim, name='dec_in')
    deco = autoencoder.layers[-7](input_dec)
    deco = autoencoder.layers[-6](deco)
    deco = autoencoder.layers[-5](deco)
    deco = autoencoder.layers[-4](deco)
    deco = autoencoder.layers[-3](deco)
    deco = autoencoder.layers[-2](deco)
    deco = autoencoder.layers[-1](deco)

    # create the decoder model
    decoder = Model(input_dec, deco)

    return (autoencoder, encoder, decoder)

def getLeCunhaNet(inputShape):
    """
    the lecunha network codes from 200*200*3 to a 25*25*16 code
    :param inputShape: tuple - the shape of the input images
    :return: tuple (model, encodermap, decodermap)
    """

    input_img = Input(shape=inputShape, name='input_layer')
    x = Conv2D(16,(3,3),activation='relu',padding='same', name='enc_conv1')(input_img)
    x = Conv2D(16,(3,3),activation='relu',padding='same', name='enc_conv2')(x)
    x = MaxPooling2D((4,4), padding='same', name='enc_max_pool1')(x)
    x = Conv2D(16,(3,3),activation='relu',padding='same', name='enc_conv3')(x)
    encoded = MaxPooling2D((2,2),padding='same', name='enc_max_pool2')(x)


    x = UpSampling2D((2,2), name='dec_up_samp1')(encoded)
    x = Conv2D(16,(3,3),activation='relu', padding='same', name='dec_conv1')(x)
    x = UpSampling2D((4,4),name='dec_up_samp2')(x)
    x = Conv2D(16,(3,3),activation='relu',padding='same',name='dec_conv2')(x)
    decoded = Conv2D(3,(3,3),activation='sigmoid', padding='same', name='dec_conv3')(x)


    # Generate models
    autoencoder = Model(input_img, decoded)
    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)
    # create a placeholder for an encoded (32-dimensional) input
    encoding_dim = autoencoder.get_layer('enc_max_pool2').output_shape[1:]

    input_dec = Input(shape=encoding_dim, name='dec_in')
    deco = autoencoder.layers[-5](input_dec)
    deco = autoencoder.layers[-4](deco)
    deco = autoencoder.layers[-3](deco)
    deco = autoencoder.layers[-2](deco)
    deco = autoencoder.layers[-1](deco)

    # create the decoder model
    decoder = Model(input_dec, deco)
    return (autoencoder, encoder, decoder)
