# the top level functions for classification

def buildModels(aeArch, aeModel, clfModel, optimizer, datapath=''):


    frozenModel = None
    meltedModel = None
    untrainedModel = None
    return frozenModel, meltedModel, untrainedModel

def trainModels(frozenModel,meltedModel,untrainedModel, batchSize, epochs, datapath=''):

    frozenModel_OBJ = None
    meltedModel_OBJ = None
    untrainedModel_OBJ = None
    return frozenModel_OBJ, meltedModel_OBJ, untrainedModel_OBJ

def saveModels(configName, frozenModel_OBJ, meltedModel_OBJ, untrainedModel_OBJ, savepath=''):

    return True