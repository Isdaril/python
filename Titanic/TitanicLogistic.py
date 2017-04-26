import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
#from utilities import predictLogistic, TrainLogisticRegression, PredictLinear, TrainLinearRegression, AddTerms, RandomSplicingTest, normalize
import utilities as utils
import time
import os


def TitanicExtractMeaning(data):
	#name: extract the title
	data['Title'] = data['Name'].str.extract(", (.*?)\.",expand = False)
	data.loc[data.Title == 'Mr', 'Title'] = 0
	data.loc[data.Title.isin(('Capt', 'Major', 'Col', 'Don', 'Sir', 'Jonkheer', 'Dr', 'Rev')), 'Title'] = 1
	data.loc[data.Title == 'Master', 'Title'] = 2
	data.loc[data.Title.isin(('Mme', 'Mrs')), 'Title'] = 3
	data.loc[data.Title.isin(('Dona', 'Lady', 'the Countess')), 'Title'] = 4
	data.loc[data.Title.isin(('Miss', 'Mlle', 'Ms')), 'Title'] = 5

	#dropping the now useless columns
	data = data.drop(['Ticket', 'Name', 'PassengerId'],axis=1)
	#sex
	data.loc[data.Sex == 'male', 'Sex'] = 0
	data.loc[data.Sex == 'female', 'Sex'] = 1
	#deck
	data.Cabin.fillna(0, inplace = True)
	data.loc[data.Cabin.str[0] == 'A', 'Cabin'] = 1
	data.loc[data.Cabin.str[0] == 'B', 'Cabin'] = 2
	data.loc[data.Cabin.str[0] == 'C', 'Cabin'] = 3
	data.loc[data.Cabin.str[0] == 'D', 'Cabin'] = 4
	data.loc[data.Cabin.str[0] == 'E', 'Cabin'] = 5
	data.loc[data.Cabin.str[0] == 'F', 'Cabin'] = 6
	data.loc[data.Cabin.str[0] == 'G', 'Cabin'] = 7
	data.loc[data.Cabin.str[0] == 'T', 'Cabin'] = 8
	#embarked
	data.Embarked.fillna(0, inplace = True)
	data.loc[data.Embarked == 'C', 'Embarked'] = 1
	data.loc[data.Embarked == 'Q', 'Embarked'] = 2
	data.loc[data.Embarked == 'S', 'Embarked'] = 3
	#fare
	data.Fare.fillna(data.Fare[data.Pclass == 3].mean(), inplace = True)
	#age: to fill in the missing ages, we are going to use a linear regression
	AgePrediction = TrainAgeRegresssion(data)

	AgePrediction = pd.DataFrame(AgePrediction, columns=['Age'])
	data.Age.fillna(AgePrediction.Age, inplace = True)
	return data


def TrainAgeRegresssion(data):
	print('training age regression')
	set = data.loc[np.logical_not(pd.isnull(data.Age))]
	results = set.as_matrix(['Age'])
	set = set.drop(['Survived', 'Age'],axis=1)
	train = np.array(set.as_matrix())
	train = train.astype(float)
	#add cubic terms
	#NB : I did a mini investigation to find a good model for age and cubic terms seem quite good
	train = utils.AddTerms(train, 3)
	#normalize the data
	(train, norm) = utils.normalize(train)
	#adding the bias
	(m,n) = train.shape
	train = np.c_[np.ones(m),train]
	#Now let's train our model
	alpha = 0.1
	lamb = 0.05
	nbiter = 1000
	drawhisto = False
	weights = utils.TrainLinearRegression(train, results, alpha, lamb, nbiter, drawhisto)
	#now predict the data
	data = data.drop(['Age', 'Survived'],axis=1)
	data = np.array(data.as_matrix())
	data = data.astype(float)
	data = utils.AddTerms(data, 3)
	data = data/norm
	(m,n) = data.shape
	data = np.c_[np.ones(m),data]
	prediction = np.absolute(utils.PredictLinear(data, weights))
	return prediction


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
alpha = 1
lamb = 0
nbiter = 100
drawhisto = False
level = 1
splicing = 20
alphaSet = [0.5,1,1.5,2,5]
lambdaSet = [0.05,0.07,0.1,0.13,0.15,0.2]
utils.RandomSplicingTest(traindata, results, splicing, nbiter, alpha, lamb, drawhisto, level)
#utils.PlotDifferentLambdas(traindata, results, lambdaSet, alpha, nbiter, N, level)
os.system('pause')
#Adding more terms
traindata = utils.AddTerms(traindata,2)
#normalize the data
(traindata, norm) = utils.normalize(traindata)
#adding the bias
(m,n) = traindata.shape
traindata = np.c_[np.ones(m),traindata]
#Now let's train our model


(weights,costs) = utils.TrainLogisticRegression(traindata, results, alpha, lamb, nbiter, drawhisto)
#now let's examine the performance of our model
#on our training set
prediction = utils.predictLogistic(traindata, weights)
errors = []
errortest = 0
for i in range(891):
	if prediction[i,0] != results[i,0]:
		errors.append(i)
		errortest += 1
DfErrors = combi[0:891].ix[errors]
DfErrors.Survived = DfErrors.Survived.astype(int)
DfErrors.to_csv('errors.csv', index=False)

print('errtest :', errortest/891 )
error = np.sum(np.square(results - prediction))/891
print('Error on training set:', error)


######################
#VALIDATION procedure
#crossval = combi[641:891]
#crossvalresults = crossval.as_matrix(['Survived'])
#del crossval['Survived']
#crossvaldat = np.array(crossval.as_matrix())
#crossvaldat = AddQuinticTerms(crossvaldat)
#crossvaldat = crossvaldat/norm
#(m,n) = crossvaldat.shape
#crossvaldat = np.c_[np.ones(m),crossvaldat]
#prediction = predictLogistic(crossvaldat, weights)
#print(prediction.shape)
#error = np.sum(np.square(crossvalresults - prediction))/250
#print('Error on validation set:', error)
#print(trainori.Survived[691:711])
#print(prediction[0:20])
######################

#testdat = np.array(test.as_matrix())
#testdat = testdat/norm
#prediction = predictLogistic(testdat, weights)
#print(prediction[0:20])
test = combi[891:]
del test['Survived']
testdat = np.array(test.as_matrix())
testdat = utils.AddTerms(testdat,2)
testdat = testdat/norm
(m,n) = testdat.shape
testdat = np.c_[np.ones(m),testdat]
prediction = utils.predictLogistic(testdat, weights)
prediction = pd.DataFrame(prediction, columns=['Survived'])
#DataPrediction  = pd.concat([trainori,testori])
#DataPrediction.Survived.fillna(prediction.Survived.astype(int), inplace = True)
#DataPrediction = DataPrediction[['PassengerId','Survived']][891:]
#DataPrediction.to_csv('LogRegression.csv', index=False)

pdtest = pd.DataFrame({'PassengerId': testori.PassengerId.astype(int), 'Survived': prediction.Survived.astype(int)})
pdtest.to_csv('LogRegression.csv', index=False)
