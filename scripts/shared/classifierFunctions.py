# the top level functions for classification
from shared.autoencoderHelpers import read_n_images, generate_img_from_folder, get_input_shape, get_num_examples, plot_history, get_images, bgr2rgb, plot_reconstruction
from shared.classifierHelpers import getUntrained, getFrozen, getMelted, getLabelShape, autoencoderToClassifier
from shared.autoencoderArchitectures import getCodeLabel

def buildModels(aeArch, aeModel, clfModel, optimizer, datapath='', modelpath=''):


    inputShape = get_input_shape(datapath,'training')
    untrainedAutoencoder = getUntrained(aeArch)
    frozenAutoencoder = getFrozen(aeModel, modelpath)
    meltedAutoencoder = getMelted(aeModel, modelpath)

    codeLabel = getCodeLabel(aeArch)
    outputShape = getLabelShape(datapath)
    frozenModel = autoencoderToClassifier(frozenAutoencoder,clfModel,codeLabel,outputShape)
    meltedModel = autoencoderToClassifier(meltedAutoencoder,clfModel,codeLabel,outputShape)
    untrainedModel = autoencoderToClassifier(untrainedAutoencoder,clfModel,codeLabel,outputShape)

    return frozenModel, meltedModel, untrainedModel

def trainModels(frozenModel,meltedModel,untrainedModel, batchSize, epochs, datapath=''):

    frozenModel_OBJ = None
    meltedModel_OBJ = None
    untrainedModel_OBJ = None
    return frozenModel_OBJ, meltedModel_OBJ, untrainedModel_OBJ

def saveModels(configName, frozenModel_OBJ, meltedModel_OBJ, untrainedModel_OBJ, savepath=''):

    return True