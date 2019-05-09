# splits and preprocesses data according to set configurations

import os, sys, pickle
import numpy as np 
import matplotlib.pyplot as plt
import shutil
from skimage import io
from skimage.transform import resize
from  shared.confReader import confReader
from  shared.statistics import confusionMatrix, countLabels
from  shared.utils import printProgressBar


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

trainingLabels = filteredLabels[:trainingSize]
validationLabels = filteredLabels[trainingSize:validationSize+trainingSize]
testLabels = filteredLabels[validationSize+trainingSize:validationSize+trainingSize+testSize]

print('classes ',imageClasses)
confMatrix = confusionMatrix([item[1] for item in trainingLabels],imageClasses,plot=True,figname='Training')
labelCounts = countLabels( [item[1] for item in trainingLabels])
print('training ', labelCounts)
confMatrix = confusionMatrix([item[1] for item in validationLabels],imageClasses,plot=True,figname='Validation')
labelCounts = countLabels([item[1] for item in validationLabels])
print('validation ',labelCounts)
confMatrix = confusionMatrix([item[1] for item in testLabels],imageClasses,plot=True, figname='Test')
labelCounts = countLabels([item[1] for item in testLabels])
print('testing ',labelCounts)

#======================prompt user if happy with distribution===
response = input('Continue with data generation? (y/n)  ')
if not response =='y':
    print('Not continuing with data generation...')
    sys.exit()

#========= generate images ====================================
def copyAndResizeImages(labels,writePath,readPath,imgSize, progress_title='progress'):
    if os.path.isdir(writePath):
        shutil.rmtree(writePath)
        os.mkdir(writePath)
    else:
        os.mkdir(writePath)

    for (i,item) in enumerate(labels):
        image = io.imread(readPath+item[0]+'.jpg')
        resized = (255*resize(image, (imgSize,imgSize,3))).astype('uint8')
        io.imsave(writePath+item[0]+'.jpg', resized)
        printProgressBar(i,len(labels),prefix=progress_title)
    


copyAndResizeImages(trainingLabels,'../data/training/',rawDataPath+'JPEGImages/',imageSize, progress_title='training extraction')
copyAndResizeImages(validationLabels,'../data/validation/',rawDataPath+'JPEGImages/',imageSize, progress_title='validation extraction')
copyAndResizeImages(testLabels,'../data/test/',rawDataPath+'JPEGImages/',imageSize, progress_title='test extraction')

splitLabels = {
    'training':trainingLabels,
    'test':testLabels,
    'validation':validationLabels,
    'classes':imageClasses
}

with open('../data/labels.pickle', 'wb' ) as f:
    pickle.dump(splitLabels, f)

#============== summarise the information for reproduction===============================
with open("../data/summary.txt","w") as f:
    f.write('classes: '+str(imageClasses).strip('[]') +'\n')
    f.write('imageSize: '+str(imageSize)+'\n')
    f.write('train-valid-test: '+str(trainingSplit)+'-'+str(validationSplit)+'-'+str(testSplit) + '\n')
    f.write('numImages:'+str(numImages)+'\n') 