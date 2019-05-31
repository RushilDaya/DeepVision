import sys
import yaml
import shared.segmentationFunctions as sg

if len(sys.argv) < 1:
    print('Please Name a configuration to run ie config1: configurations are defined in deepVision/configSegmenters.yml')
    sys.exit()

configFile = '../configSegmenters.yml'
datapath = '../data'
savepath = '../models/segmenters'

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
EPOCHS = configuration['epochs']
LOSS_FUNCTION = configuration['lossFunction']
SEGMENTATION_ARCHITECTURE = configuration['segmentationArch']

(model) = sg.buildModel(SEGMENTATION_ARCHITECTURE,OPTIMIZER,datapath=datapath)
(trainedModel) = sg.trainModel(model,BATCH_SIZE,datapath=datapath)
sg.saveModel(configName, trainedModel, savepath=savepath)