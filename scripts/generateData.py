# splits and preprocesses data according to set configurations

import os, sys
import numpy as np 
import matplotlib.pyplot as plt
from lxml import etree
from skimage import io
from skimage.transform import resize
from  shared.confReader import confReader
from shared.statistics import confusionMatrix, countLabels


#=== validate configurations ====================================

rawDataPath = confReader('RAW_DATA_LOCATION')
if not os.path.isdir(rawDataPath):
    print('please download raw data and unpack in root folder..')
    sys.exit()

imageSize=confReader('IMAGE_SIZE')
upperLimitImages=confReader('NUM_IMAGES')
trainingSplit=confReader('TRAINING_RATIO')
validationSplit=confReader('VALIDATION_RATIO')
testSplit=confReader('TEST_RATIO')
if not testSplit+trainingSplit+validationSplit == 1.0:
    print('data split doesnt add to 1')
    sys.exit()

imageClasses = confReader('CLASSES')
assert type(imageClasses) == type([])

#================================get relevant labels============

allImageNames = []
with open(rawDataPath+'ImageSets/Main/trainval.txt') as file:
    allImageNames = file.read().splitlines()

itemContainingImages = {}
for imageClass in imageClasses:
    classList = []
    with open(rawDataPath+'ImageSets/Main/'+imageClass+'_trainval.txt') as file:
        classLabels = file.read().splitlines()
        for row in classLabels:
            if row[-2:] == ' 1':
                [fileName,_,_] = row.split(' ')
                classList.append(fileName)
    itemContainingImages[imageClass] = classList

filteredLabels = []
for item in allImageNames:
    contains = [0]*len(imageClasses)
    for (i, objClass) in enumerate(imageClasses):
        if  item in itemContainingImages[objClass]:
            contains[i] = 1
    if sum(contains) > 0:
        filteredLabels.append((item, contains))

labelsOnly = [item[1] for item in filteredLabels]

    
#========= build the datasets ==================================
numImages = len(filteredLabels)
if upperLimitImages != -1:
    numImages = min(numImages, upperLimitImages)
    
trainingSize = int(trainingSplit*numImages)
validationSize = int(validationSplit*numImages)
testSize = int(testSplit*numImages)

trainingLabels = labelsOnly[:trainingSize]
validationLabels = labelsOnly[trainingSize:validationSize+trainingSize]
testLabels = labelsOnly[validationSize+trainingSize:validationSize+trainingSize+testSize]

print('classes ',imageClasses)
confMatrix = confusionMatrix(trainingLabels,imageClasses,plot=True,figname='Training')
labelCounts = countLabels(trainingLabels)
print('training ', labelCounts)
confMatrix = confusionMatrix(validationLabels,imageClasses,plot=True,figname='Validation')
labelCounts = countLabels(validationLabels)
print('validation ',labelCounts)
confMatrix = confusionMatrix(testLabels,imageClasses,plot=True, figname='Test')
labelCounts = countLabels(testLabels)
print('testing ',labelCounts)

#======================prompt user if happy with distribution===
response = input('Continue with data generation? (y/n)  ')
if not response =='y':
    print('Not continuing with data generation...')
    sys.exit()

#========= generate pickles ====================================
