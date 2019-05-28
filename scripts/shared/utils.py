import os, shutil

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '*'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r %s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def createUpdateDirectories(directoryArray, reset=False):
    """
    creates directories if they dont exist
    if they do exist and reset is False nothing happens
    if they do exist and reset is True the directory is cleared and reset
    """
    for directory in directoryArray:
        if os.path.isdir(directory):
            if reset:
                shutil.rmtree(directory)
                os.mkdir(directory)
        else:
            os.mkdir(directory)
    
    return True