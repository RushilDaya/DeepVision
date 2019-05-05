import yaml

FILE_PATH = '../configVars.yml'

def confReader(configItem):
    with open(FILE_PATH, 'r') as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    configuration = configs[configItem]
    return configuration
