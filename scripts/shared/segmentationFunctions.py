import numpy as np
from math import ceil
from random import randint
import pickle, os
import tensorflow as tf
from shared.segmentationHelpers import get_input_shape, get_num_images, generate_image_segmentation_labels
from shared.segmentationArchitectures import miniUnet, midiUnet

def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred)
  # some implementations don't square y_pred
  denominator = tf.reduce_sum(y_true + tf.square(y_pred))

  return 1 - numerator / (denominator + tf.keras.backend.epsilon())

def buildModel(segmentationArchitecture, optimizer,lossFunction,segmentationScheme,datapath=''):
    inputShape = get_input_shape(datapath,segmentationScheme)
    
    if segmentationArchitecture == 'miniUnet':
        model = miniUnet(inputShape)
    elif segmentationArchitecture == 'midiUnet':
        model = midiUnet(inputShape)
    else:
        raise TypeError('undefined architecure')
    # need to add the DICE metric

    if lossFunction == 'dice':
        print('using DICE loss')
        lossFunction = dice_loss

    model.compile(optimizer=optimizer,loss=lossFunction)
    return (model)

def trainModel(model, batchSize, epochs, segmentationScheme, datapath=''):
    num_samples_train = get_num_images('training',segmentationScheme,datapath)
    steps_train = ceil(num_samples_train/batchSize)
    num_samples_validation = get_num_images('validation',segmentationScheme,datapath)
    print('****')
    print(segmentationScheme)
    steps_validation = ceil(num_samples_validation/batchSize)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit_generator(generate_image_segmentation_labels('training',segmentationScheme ,batchSize, dataDir=datapath,squashOutput=True),
                                    shuffle=True,
                                    validation_data=generate_image_segmentation_labels('validation',segmentationScheme , batchSize, dataDir=datapath,squashOutput=True),
                                    steps_per_epoch=steps_train, validation_steps=steps_validation,
                                    epochs=epochs)

    return (model,history)

def saveModel(configName, model, history, segmentationScheme, savepath=''):

    savePathPartial = savepath+'/'+segmentationScheme+'/'
    if not os.path.isdir(savePathPartial):
        os.mkdir(savePathPartial)

    saveLoc = savePathPartial +configName+'/'
    if not os.path.isdir(saveLoc):
        os.mkdir(saveLoc)
    tf.keras.models.save_model(model, saveLoc+'model', overwrite=True)

    with open(saveLoc+'history.pickle', 'wb' ) as f:
        pickle.dump(history.history, f)

    return None