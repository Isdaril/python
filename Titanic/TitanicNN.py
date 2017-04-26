import pandas as pd
import numpy as np
import utilities as utils
from TitanicUtils import TitanicExtractMeaning
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras import regularizers
import math

def NNModel(inputShape, weights_path=None):
    model = Sequential()
    #model.add(Flatten())
    model.add(Dense(600, activation='sigmoid',input_shape=inputShape))
    model.add(Dropout(0.5))
    #model.add(Dense(300, activation='sigmoid'))
    #model.add(Dropout(0.5))
    #model.add(Dense(100, activation='sigmoid'))
    #model.add(Dropout(0.5))
    #model.add(Dense(60, activation='sigmoid'))
    #model.add(Dropout(0.5))
    #model.add(Dense(20, activation='sigmoid'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    if weights_path:
        model.load_weights(weights_path)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

trainori = pd.read_csv('train.csv')
testori = pd.read_csv('test.csv')
#merge the data and work with it (fill the gaps, extract some useful information)
testori["Survived"] = np.nan
combi = pd.concat([trainori,testori])
combi = TitanicExtractMeaning(combi)
#now that we are happy with our data, let's get back our training and testing sets
train = combi[0:891]
results = train.as_matrix(['Survived'])
del train['Survived']
traindata = np.array(train.as_matrix())
traindata = traindata.astype(float)
(m,n) = traindata.shape
model = NNModel((n,),weights_path=None)
N = 2
batchSize = 32
epochs = 500
drawhisto = True
utils.RandomSplicingNN(traindata, results, N, model, batchSize, epochs,drawhisto)
