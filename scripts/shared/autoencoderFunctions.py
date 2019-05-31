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
from shared.autoencoderArchitectures import getNormal, getDayaNet, getLeCunhaNet



def buildModel(networkArch='normal', optimizer='adam', datapath=''):
    """
    compiles one of the predefined network architectures with a specific optimizer
    :param networkArch: string - must be a predefined architecure name
    :param optimizer: string - any valid keras optimizer
    :param datapath: string - path to datafolder used to inspect image size
    :return: tuple (model, encoder, decoder)
    """

    in_shape = get_input_shape(datapath, 'training')
    if networkArch =='normal':
        (autoencoder,encoder,decoder)=getNormal(in_shape)
    elif networkArch == 'dayaNet':
        (autoencoder,encoder,decoder)=getDayaNet(in_shape)
    elif networkArch == 'leCunhaNet':
        (autoencoder,encoder,decoder)=getLeCunhaNet(in_shape)
    else:
        raise TypeError('architecture not implemented')

    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['binary_crossentropy'])
    encoder.summary()
    decoder.summary()
    autoencoder.summary()

    return (autoencoder,encoder,decoder)

def trainModel(autoencoder, batchsize=32, epochs=1, datapath=''):
    """
    trains the model autoencoder with all the data located at datapath
    :param autoencoder: keras model - an keras model
    :param batchsize: int - minibatch size
    :param epochs: int - num epochs
    :param datapath: string - location of data, expects to find a training and validation subfolder
    :return: tuple (autoencoder, historyObject)
    """
    
    DATA_DIR = datapath
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

    return (autoencoder,history)


def saveModel(configName, nnModel, history, savepath=''):
    """
    saves the model and history in a folder named configName located at the savepath
    :param configName: string -
    :param nnModel: model -
    :param history: kerasObject -
    :param savepath: string -
    """
    saveLoc = savepath+'/'+configName+'/'
    if not os.path.isdir(saveLoc):
        os.mkdir(saveLoc)
    tf.keras.models.save_model(nnModel, saveLoc+'model', overwrite=True)

    with open(saveLoc+'history.pickle', 'wb' ) as f:
        pickle.dump(history.history, f)

    return True



