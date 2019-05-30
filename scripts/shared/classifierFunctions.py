# the top level functions for classification
from shared.autoencoderHelpers import read_n_images, generate_img_from_folder, get_input_shape, get_num_examples, plot_history, get_images, bgr2rgb, plot_reconstruction
from shared.classifierHelpers import getUntrained, getFrozen, getMelted, get_num_classes, autoencoderToClassifier, generate_img_label_from_folder
from shared.autoencoderArchitectures import getCodeLabel

import numpy as np
from math import ceil
from random import randint
import pickle, os
import tensorflow as tf

def buildModels(aeArch, aeModel, clfModel, optimizer, datapath='', modelpath=''):


    inputShape = get_input_shape(datapath,'training')
    (untrainedAutoencoder,_,_) = getUntrained(aeArch, inputShape)
    frozenAutoencoder = getFrozen(aeModel, modelpath)
    meltedAutoencoder = getMelted(aeModel, modelpath)

    codeLabel = getCodeLabel(aeArch)
    outputShape = get_num_classes(datapath)
    frozenModel = autoencoderToClassifier(frozenAutoencoder,clfModel,codeLabel,outputShape)
    meltedModel = autoencoderToClassifier(meltedAutoencoder,clfModel,codeLabel,outputShape)
    untrainedModel = autoencoderToClassifier(untrainedAutoencoder,clfModel,codeLabel,outputShape)

    frozenModel.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
    meltedModel.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
    untrainedModel.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])

    return frozenModel, meltedModel, untrainedModel

def trainModels(frozenModel,meltedModel,untrainedModel, batchSize, epochs, datapath=''):

    NUM_SAMPLES_TRAIN = get_num_examples(datapath, 'training')
    STEPS_PER_EPOCH = ceil(NUM_SAMPLES_TRAIN/batchSize)

    NUM_SAMPLES_VAL = get_num_examples(datapath, 'validation')
    VALIDATION_STEPS=ceil(NUM_SAMPLES_VAL/batchSize)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    historyFrozen = frozenModel.fit_generator(generate_img_label_from_folder(datapath, 'training', batchSize), shuffle=True,
                                    validation_data=generate_img_label_from_folder(datapath, 'validation', batchSize),
                                    steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS,
                                    epochs=epochs)
    frozenModel_OBJ = (frozenModel, historyFrozen)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    historyMelted = meltedModel.fit_generator(generate_img_label_from_folder(datapath, 'training', batchSize), shuffle=True,
                                    validation_data=generate_img_label_from_folder(datapath, 'validation', batchSize),
                                    steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS,
                                    epochs=epochs)
    meltedModel_OBJ = (meltedModel, historyMelted)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    historyUntrained = untrainedModel.fit_generator(generate_img_label_from_folder(datapath, 'training', batchSize), shuffle=True,
                                    validation_data=generate_img_label_from_folder(datapath, 'validation', batchSize),
                                    steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS,
                                    epochs=epochs)
    untrainedModel_OBJ = (untrainedModel, historyUntrained)

    return frozenModel_OBJ, meltedModel_OBJ, untrainedModel_OBJ

def saveModels(configName, frozenModel_OBJ, meltedModel_OBJ, untrainedModel_OBJ, savepath=''):

    baseDir = savepath + '/'+configName +'/'
    saveLocFrozen = baseDir + 'frozen/'
    saveLocMelted = baseDir + 'melted/'
    saveLocUntrained = baseDir + 'untrained/'

    if not os.path.isdir(baseDir):
        os.mkdir(baseDir)
    if not os.path.isdir(saveLocFrozen):
        os.mkdir(saveLocFrozen)
    if not os.path.isdir(saveLocMelted):
        os.mkdir(saveLocMelted)
    if not os.path.isdir(saveLocUntrained):
        os.mkdir(saveLocUntrained)   

    tf.keras.models.save_model(frozenModel_OBJ[0], saveLocFrozen+'model', overwrite=True)
    tf.keras.models.save_model(meltedModel_OBJ[0], saveLocMelted+'model', overwrite=True)
    tf.keras.models.save_model(untrainedModel_OBJ[0], saveLocUntrained+'model', overwrite=True)         

    with open(saveLocFrozen+'history.pickle', 'wb' ) as f:
        pickle.dump(frozenModel_OBJ[1].history, f)
    with open(saveLocMelted+'history.pickle', 'wb' ) as f:
        pickle.dump(meltedModel_OBJ[1].history, f)
    with open(saveLocUntrained+'history.pickle', 'wb' ) as f:
        pickle.dump(untrainedModel_OBJ[1].history, f)

    return True