# performs the classifier training
# the input configuration is trained in three ways 

import sys
import yaml
import shared.classifierFunctions as clf

if len(sys.argv) < 1:
    print('Please Name a configuration to run ie config1: configurations are defined in deepVision/configClassifiers.yml')
    sys.exit()

configFile = '../configClassifiers.yml'
datapath = '../data'
savepath = '../models/classifiers'

configName = sys.argv[1]

with open(configFile, 'r') as stream:
    try:
        configs = yaml.safe_load(stream)
        configuration = configs[configName]
    except:
        print('an error occurred in loading the configuration')
        sys.exit()


OPTIMIZER = configuration['optimizer']
BATCH_SIZE = configuration['batchSize']
EPOCHS =  configuration['epochs']
AUTOENCODER_ARCHITECTURE = configuration['autoencoderArch'] 
AUTOENCODER_MODEL = configuration['autoencoderModel']
CLASSIFIER_MODEL = configuration['classifierModel']

(frozenModel, meltedModel ,untrainedModel) = clf.buildModels(AUTOENCODER_ARCHITECTURE, AUTOENCODER_MODEL,CLASSIFIER_MODEL, OPTIMIZER, datapath=datapath)
(frozenModel_OBJ, meltedModel_OBJ,untrainedModel_OBJ) = clf.trainModels(frozenModel,meltedModel,untrainedModel, BATCH_SIZE, EPOCHS, datapath=datapath)
clf.saveModels(configName, frozenModel_OBJ, meltedModel_OBJ ,untrainedModel_OBJ, savepath=savepath )
