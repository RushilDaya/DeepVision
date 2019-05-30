# low level functions used primarily in the classifier
import tensorflow as tf
import pickle
import glob
import cv2
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from shared.autoencoderHelpers import read_n_images
from shared.autoencoderArchitectures import getNormal, getDayaNet, getLeCunhaNet 
from shared.classifierArchitectures import getSimple

def getUntrained(aeArchitecture, inputShape):

    if aeArchitecture == 'normal':
        model = getNormal(inputShape)
    elif aeArchitecture == 'dayaNet':
        model = getDayaNet(inputShape)
    elif aeArchitecture == 'leCunhaNet':
        model = getLeCunhaNet(inputShape)
    else:
        raise TypeError('undefined Architecture')

    return model

def getFrozen(aeModelName, modelpath):
    fullPath = modelpath + '/autoencoder/'+aeModelName+'/model'
    model = tf.keras.models.load_model(fullPath)

    for layer in model.layers:
        layer.trainable = False 

    return model

def getMelted(aeModelName, modelpath):
    fullPath = modelpath + '/autoencoder/'+aeModelName+'/model'
    model = tf.keras.models.load_model(fullPath)

    for layer in model.layers:
        layer.trainable = True

    return model

def get_num_classes(data_dir):
    """
    Get the number of classes.
    :param data_dir: str - data directory
    :return: int - number of classes
    """
    mode = 'training' # arbitrary
    loc = "{}/{}".format(data_dir, mode)
    with open('{}/labels.pickle'.format(data_dir), 'rb') as f:
        data = pickle.load(f)

    modes = list(data.keys())

    assert glob.glob(data_dir), "Check directory."
    assert glob.glob("{}/*.jpg".format(loc)), "Check file extension (should be 'jpg')."
    i = 0  # Arbitrarily chosen
    return len(data[modes[i]][i][-1])

def autoencoderToClassifier(autoencoderModel, classifierName, codeName, outputShape):

    layerNames = [layer.name for layer in autoencoderModel.layers]
    encodedLayerId = layerNames.index(codeName)
    encoder = Model(inputs=autoencoderModel.inputs, outputs=autoencoderModel.layers[encodedLayerId].output)

    if classifierName == 'simple':
        classifierOutput = getSimple(encoder, outputShape)
    else:
        raise TypeError('classifier Not implemented')

    classifier = Model(inputs=encoder.inputs, outputs=classifierOutput)
    return classifier

def generate_img_label_from_folder(data_dir, mode, batch_size):
    """

    """
    loc = "{}/{}".format(data_dir, mode)
    while True:
        with open('{}/labels.pickle'.format(data_dir), 'rb') as f:
            data = pickle.load(f)

        modes = list(data.keys())
        del modes[-1]

        assert mode in modes, "'{}' not a valid mode (must be one of {})".format(mode, str(modes))
        assert glob.glob(loc), "Check directory."
        assert glob.glob("{}/*.jpg".format(loc)), "Check file extension (should be 'jpg')."
        for idx in range(0, len(data[mode]), batch_size):
            start = idx
            end = idx + batch_size
            images, labels = read_n_images(data[mode], start, end, loc)
            yield (images/255, np.array(labels))