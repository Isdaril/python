import os
import random
import math
import shutil

pathFish = r'D:\Documents\Cours\Python\Fishy\data\Fish'
nameFish = 'fish'
pathNotAFish = r'D:\Documents\Cours\Python\Fishy\data\NotAFish'
nameNotAFish = 'notAFish'
extension = '.jpg'

def PrepareSets(path, nameOfFiles, extension):
    #rename the files and randomze them then arrange them in crossval and training sets
    validationPath = path + r'\validation'
    trainPath = path + r'\train'
    files = os.listdir(path)
    i = 1
    n = len(files)
    crossValSize = math.floor(n/3)
    for theFile in files:
        os.rename(os.path.join(path, theFile), os.path.join(path, 'tmp_'+str(i).zfill(4)+extension))
        i = i+1
    i=1
    files = os.listdir(path)

    if not os.path.exists(validationPath):
        os.makedirs(validationPath)
    if not os.path.exists(trainPath):
        os.makedirs(trainPath)
    while (len(files) > 0):
        #rearrange the files randomly
        theFile = random.choice(files)
        files.remove(theFile)
        if i <= crossValSize:
            shutil.move(os.path.join(path, theFile), os.path.join(validationPath, nameOfFiles + '_' + str(i).zfill(4) + extension))
        else:
            shutil.move(os.path.join(path, theFile), os.path.join(trainPath, nameOfFiles + '_' + str(i).zfill(4) + extension))
        i = i+1
    i = 1
    files = os.listdir(path)


PrepareSets(pathFish, nameFish, extension)
#PrepareSets(pathNotAFish, nameNotAFish, extension)
