# functions which generate summary statistics
import matplotlib.pyplot as plt
import numpy as np

def confusionMatrix(instances, labels, plot=False, figname=''):
    # instances is a list of lists
    # each contained list has N values (type: 0 || 1) 
    # returns a list of lists (N by N) which shows the co-occurance of 1's
    
    def _sum(listA,listB):
        assert len(listA) == len(listB)
        return [val + listB[idx] for (idx, val) in enumerate(listA)]


    N = len(instances[0])
    confMat = [[0]*N]*N

    for classLabel in range(N):
        reducedInstances = [item for item in instances if item[classLabel]==1]
        for item in reducedInstances:
            confMat[classLabel] = _sum(confMat[classLabel], item)

    if plot:
        fig, ax = plt.subplots()
        im = ax.imshow(confMat)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_title(figname)
        plt.show()

    return confMat

def countLabels(instances):
    # column wise sums on a list of lists
    # rather use numpy arrays where possible instead of needing this function

    itemCounts = [0]*len(instances[0])
    for row in instances:
        for item in range(len(itemCounts)):
            itemCounts[item]+=row[item]
    return itemCounts
