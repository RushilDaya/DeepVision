import sys
import yaml
from helper_models.autoencoderTopLevels import AEsetup, AEtrain, AEsave

if len(sys.argv) < 1:
    print('Please Name a configuration to run ie config1')
    sys.exit()

configFile = 'autoencoderConfigs.yml'
configName = sys.argv[1]

with open(configFile, 'r') as stream:
    try:
        configs = yaml.safe_load(stream)
        configuration = configs[configName]
    except:
        print('an error occurred in loading the configuration')
        sys.exit()

(model,encoderPart,decoderPart)=AEsetup(configuration['network'],configuration['optimizer'])
(model,history)=AEtrain(model, configuration['batchSize'],configuration['epochs'])
AEsave(configName,model,history)


