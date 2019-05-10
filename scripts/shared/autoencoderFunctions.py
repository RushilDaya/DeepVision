from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from random import randint
import os
from shared.autoencoderHelpers import read_n_images, generate_img_from_folder, get_input_shape, get_num_examples, plot_history, get_images, bgr2rgb, plot_reconstruction




def buildModel(networkArch='normal', optimizer='adam'):
    DATA_DIR = '../data'
    in_shape = get_input_shape(DATA_DIR, 'training')
    input_img = Input(shape=in_shape, name='input_layer')

    if networkArch =='normal':
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

    if networkArch == 'dayaNet':
        # deeper than the normal case with a much tighter bottleneck (code space of 5 by 5 by 4)
        # expect worse performance than normal but with a much smaller code :)

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
    
    if networkArch == 'leCunhaNet':
        # network has a larger coding space than the normal network 
        # should perform better in terms of reconstruction accuracy though

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

    else:
        raise TypeError('architecture not implemented')


    autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['mean_squared_error'])
    encoder.summary()
    decoder.summary()
    autoencoder.summary()
    return autoencoder,encoded,decoded

def trainModel(autoencoder, batchsize=32, epochs=1, datapath=''):
    
    DATA_DIR = '../data'
    EPOCHS = epochs
    BATCH_SIZE_TRAIN = batchsize
    NUM_SAMPLES_TRAIN = get_num_examples(DATA_DIR, 'training')
    STEPS_PER_EPOCH = ceil(NUM_SAMPLES_TRAIN/BATCH_SIZE_TRAIN)

    BATCH_SIZE_VAL= batchsize
    NUM_SAMPLES_VAL = get_num_examples(DATA_DIR, 'validation')
    VALIDATION_STEPS=ceil(NUM_SAMPLES_VAL/BATCH_SIZE_VAL)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = autoencoder.fit_generator(generate_img_from_folder(DATA_DIR, 'training', BATCH_SIZE_TRAIN), shuffle=True,
                                        validation_data=generate_img_from_folder(DATA_DIR, 'validation', BATCH_SIZE_VAL),
                                        steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS,
                                        epochs=EPOCHS)

    return autoencoder,history


def saveModel(configName, nnModel, history, savepath=''):
    saveLoc = savepath+'/'+configName+'/'
    if not os.path.isdir(saveLoc):
        os.mkdir(saveLoc)

    tf.keras.models.save_model(nnModel, saveLoc+'model', overwrite=True)

    with open(saveLoc+'history.pickle', 'wb' ) as f:
        pickle.dump(history.history, f)

    return True



