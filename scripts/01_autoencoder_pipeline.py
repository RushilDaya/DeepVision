import sys
import yaml
import shared.autoencoderFunctions as ae

if len(sys.argv) < 1:
    print('Please Name a configuration to run ie config1: configurations are defined in deepVision/configAutoencoders.yml')
    sys.exit()

configFile = '../configAutoencoders.yml'
datapath = '../data'
savepath = '../models/autoencoder'

configName = sys.argv[1]

with open(configFile, 'r') as stream:
    try:
        configs = yaml.safe_load(stream)
        configuration = configs[configName]
    except:
        print('an error occurred in loading the configuration')
        sys.exit()


(model, encoderPart, decoderPart) = ae.buildModel(configuration['network'],configuration['optimizer'],datapath=datapath)
(model, history) = ae.trainModel(model, configuration['batchSize'],configuration['epochs'], datapath=datapath)
ae.saveModel(configName, model, history, savepath=savepath)


