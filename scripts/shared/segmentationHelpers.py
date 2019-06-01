import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from random import randint
import pickle,glob, cv2,os

def get_input_shape(dataPath, segmentationScheme):

    imgLoc = "{}/segmentation/{}/training/images/".format(dataPath,segmentationScheme)
    labelLoc = "{}/segmentation/{}/labels.pickle".format(dataPath,segmentationScheme)

    with open(labelLoc, 'rb') as f:
        data = pickle.load(f)
    data = data['training']
    imageName = data[0]
    img = cv2.imread("{}{}.jpg".format(imgLoc,imageName))
    return img.shape

def read_n_images(data, start, end, dataPath):
    """
    Read images (should be jpg) from a dataset (from indexes start to end).
    :param data: list - image names
    :param start: int - start index
    :param end: int - end index
    :param loc: str - directory location of the images
    :return: numpy - numpy array of (BGR) image
    """
    assert glob.glob(dataPath), "Check directory."
    assert glob.glob("{}/*.jpg".format(dataPath)), "Check file extension (should be 'jpg')."
    images_list = data[start:end]
    images = [cv2.imread("{}/{}.jpg".format(dataPath, image)) for image in images_list]
    return np.array(images)

def generate_image_segmentation_labels(method,segmentationScheme ,batchSize, dataDir='', squashOutput=True):
    
    imagePath =  "{}/segmentation/{}/{}/images".format(dataDir,segmentationScheme,method)
    segmentsPath = "{}/segmentation/{}/{}/labels".format(dataDir,segmentationScheme,method)
    labelPath = "{}/segmentation/{}/labels.pickle".format(dataDir,segmentationScheme)
    while True:
            with open(labelPath, 'rb') as f:
                data = pickle.load(f)

            methods = list(data.keys())

            assert method in methods, "'{}' not a valid mode (must be one of {})".format(method, str(methods))
            data = data[method]
            for idx in range(0, len(data), batchSize):
                start = idx
                end = idx + batchSize
                images = read_n_images(data, start, end, imagePath)
                segmentations = read_n_images(data, start, end, segmentsPath)
                if squashOutput == True:
                    segmentations = segmentations[:,:,:,0]+segmentations[:,:,:,1]+segmentations[:,:,:,2]
                    segmentations = segmentations
                    sShape = segmentations.shape
                    segmentations = segmentations.reshape((sShape[0],sShape[1],sShape[2],1))

                yield (images / 255, segmentations / 255)
                
def get_num_images(method, segmentationScheme, dataDir):
    labelPath = "{}/segmentation/{}/labels.pickle".format(dataDir, segmentationScheme)
    with open(labelPath, 'rb') as f:
        data = pickle.load(f)
    data = data[method]
    return len(data)