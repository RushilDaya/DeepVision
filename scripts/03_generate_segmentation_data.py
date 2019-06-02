# running this scripts produces the segmentation data sets.
import os, sys, pickle
import numpy as np 
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from  shared.confReader import confReader
from  shared.utils import printProgressBar, createUpdateDirectories

import sys
import yaml


def copyAndResizeImage(originalImagePaths,targetImagePaths,imgSize, progress_title='progress', processFunction=None):
    """
        moves images from one location to another with resizing
        if processFunction is not None then it gets applied to the image before saving
    """
    if len(originalImagePaths) != len(targetImagePaths):
        raise TypeError('invalid input arguments')

    numCopies = len(originalImagePaths)

    for i in range(numCopies):
        image = io.imread(originalImagePaths[i])
        resized = (255*resize(image, (imgSize,imgSize,3))).astype('uint8')
        if not processFunction is None:
            processed = processFunction(resized)
        else:
            processed = resized
        io.imsave(targetImagePaths[i], processed)
        printProgressBar(i, numCopies,prefix=progress_title)

def getProcesser(segmentationType):
    """
        returns a decorator function based on the segementation type we want
        the returned function takes in an image and returns the processed image
    """
    if segmentationType == 'foregroundBackground':
        def binaryForegroundBackground(image):
            (height,width,depth) = image.shape
            for i in range(height):
                for j in range(width):
                    if np.array_equal(image[i,j,:],[0,0,210]): # this comparision is based on a visualisation of the images
                        image[i,j,:] = [0,0,0]
                    else:
                        image[i,j,:] = [255,255,255]
            return image
        return binaryForegroundBackground

    if segmentationType == 'personOnly':
        def binaryPersonOnly(image):
            # figuring out the segmentation value requires a bit of trail and error
            (height,width,depth) = image.shape
            for i in range(height):
                for j in range(width):
                    if (image[i,j,0] > 177 and image[i,j,0] < 182) and (image[i,j,1] > 125 and image[i,j,1] < 130) and  (image[i,j,2] > 230 and image[i,j,2] < 235):
                        image[i,j,:] = [255,255,255]
                    else:
                        image[i,j,:] = [0,0,0]
            return image
        return binaryPersonOnly


    else:
        raise TypeError('segmentation type not defined')


def singleOutClass(allNames, classOfInterest, rawDataPath):
    with open(rawDataPath+'ImageSets/Main/{}_trainval.txt'.format(classOfInterest)) as file:
        dirtyLines = file.read().splitlines()

    classImageNames = []
    for i in range(len(dirtyLines)):
        if dirtyLines[i][-2:]==' 1':
            classImageNames.append(dirtyLines[i][:-3])

    newList = list(set(allNames).intersection(classImageNames))
    print(len(newList))
    print('---')
    return newList



if len(sys.argv) < 1:
    print('expected argument: segmentationScheme eg. foregroundBackground )
    sys.exit()

segmentationType = sys.argv[1]
if not segmentationType in ['foregroundBackground','personOnly']:
    print('invalid segmentation type in configFile')
    sys.exit()



rawDataPath = confReader('RAW_DATA_LOCATION')
if not os.path.isdir(rawDataPath):
    print('please download raw data and unpack in root folder..')
    sys.exit()

imageSize=confReader('IMAGE_SIZE')

trainingSplit=confReader('TRAINING_RATIO')
validationSplit=confReader('VALIDATION_RATIO')
testSplit=confReader('TEST_RATIO')
if not testSplit+trainingSplit+validationSplit == 1.0:
    print('data split doesnt add to 1')
    sys.exit()


storeDir = '../data/segmentation/'+segmentationType+'/'
createUpdateDirectories(['../data/segmentation/'],reset=False)
createUpdateDirectories([storeDir],reset=False)
createUpdateDirectories([storeDir+item+'/' for item in ['training','validation','test']], reset=True)
createUpdateDirectories([storeDir+item+'/' for item in ['training/images','validation/images','test/images','training/labels','validation/labels','test/labels']], reset=True)

allImageNames  = []
with open(rawDataPath+'ImageSets/Segmentation/trainval.txt') as file:
    allImageNames = file.read().splitlines()


if segmentationType == 'personOnly':
    allImageNames = singleOutClass(allImageNames,'person', rawDataPath)

numImages = len(allImageNames)
trainingNames = allImageNames[:int(numImages*trainingSplit)]
validationNames = allImageNames[int(numImages*trainingSplit):int(numImages*(trainingSplit+validationSplit))]
testNames = allImageNames[int(numImages*(trainingSplit+validationSplit)):]


splitLabels = {
    'training':trainingNames,
    'test':testNames,
    'validation':validationNames
}

with open('../data/segmentation/'+segmentationType+'/labels.pickle', 'wb' ) as f:
    pickle.dump(splitLabels, f)



trainingImagePaths = [rawDataPath+'JPEGImages/'+item+'.jpg' for item in trainingNames]
trainingLabelPaths = [rawDataPath+'SegmentationClass/'+item+'.png' for item in trainingNames]
trainingImagePathsTarget = [storeDir+'training/images/'+item+'.jpg' for item in trainingNames]
trainingLabelPathsTarget = [storeDir+'training/labels/'+item+'.jpg' for item in trainingNames]
copyAndResizeImage(trainingImagePaths,trainingImagePathsTarget,imageSize,processFunction=None)
copyAndResizeImage(trainingLabelPaths,trainingLabelPathsTarget,imageSize,processFunction=getProcesser(segmentationType))

validationImagePaths = [rawDataPath+'JPEGImages/'+item+'.jpg' for item in validationNames]
validationLabelPaths = [rawDataPath+'SegmentationClass/'+item+'.png' for item in validationNames]
validationImagePathsTarget = [storeDir+'validation/images/'+item+'.jpg' for item in validationNames]
validationLabelPathsTarget = [storeDir+'validation/labels/'+item+'.jpg' for item in validationNames]
copyAndResizeImage(validationImagePaths,validationImagePathsTarget,imageSize,processFunction=None)
copyAndResizeImage(validationLabelPaths,validationLabelPathsTarget,imageSize,processFunction=getProcesser(segmentationType))

testImagePaths = [rawDataPath+'JPEGImages/'+item+'.jpg' for item in testNames]
testLabelPaths = [rawDataPath+'SegmentationClass/'+item+'.png' for item in testNames]
testImagePathsTarget = [storeDir+'test/images/'+item+'.jpg' for item in testNames]
testLabelPathsTarget = [storeDir+'test/labels/'+item+'.jpg' for item in testNames]
copyAndResizeImage(testImagePaths,testImagePathsTarget,imageSize,processFunction=None)
copyAndResizeImage(testLabelPaths,testLabelPathsTarget,imageSize,processFunction=getProcesser(segmentationType))