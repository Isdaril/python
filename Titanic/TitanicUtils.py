import pandas as pd
import numpy as np
import utilities as utils

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
