import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def TitanicExtractMeaning(data):
	#name: extract the title
	data['Title'] = data['Name'].str.extract(", (.*?)\.",expand = False)
	data.loc[data.Title.isin(('Miss', 'Mlle', 'Ms')), 'Title'] = 0
	data.loc[data.Title == 'Master', 'Title'] = 1
	data.loc[data.Title.isin(('Mme', 'Mrs')), 'Title'] = 2
	data.loc[data.Title == 'Mr', 'Title'] = 3
	data.loc[data.Title.isin(('Dona', 'Lady', 'the Countess')), 'Title'] = 4
	data.loc[data.Title.isin(('Capt', 'Major', 'Col', 'Don', 'Sir', 'Jonkheer', 'Dr', 'Rev')), 'Title'] = 5	
	#dropping the now useless columns
	data = data.drop(['Ticket', 'Name', 'PassengerId', 'Survived'],axis=1)
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
	return data
	
def TrainLinearRegression(data, results, alpha, lamb, nbiter, drawhisto):
	#data is the normalized training set with bias added
	#results is the vector of results (containing 0 or 1)
	#alpha is the learning rate
	#lambda is the regularization constant
	#nbiter is the number of step we take to train our logistic regression
	(m,n) = data.shape
	weights = np.random.random(n) - 0.5
	weights = weights.reshape(n,1)
	#create a vector of costs
	if drawhisto:
		costs = []
	for i in range(nbiter):	
		#computing the regularized gradient (we exclude the bias term from the regularized expression)
		grad = 1/m * np.dot(np.transpose(data), np.dot(data, weights) - results) + lamb/m*np.insert(weights[1:,:],0, 0).reshape(n,1)
		weights = weights - alpha*grad

		if drawhisto:
			#compute cost
			cost = 1/2/m * np.sum(np.square(results - np.dot(data, weights)))
			print('cost: ', cost)
			costs.append(cost)
			
	#draw histo
	if drawhisto:
		plt.plot(list(range(1,nbiter+1)), costs)
		plt.show()
	return weights
	
def PredictLinear(data,weights):
	predict = np.dot(data,weights)
	return predict

def normalize(data):
	(m,n) = data.shape
	norm = np.zeros(n)
	for i in range(n):
		max = np.amax(data[:,i])
		data[:,i] = data[:,i]/max
		norm[i]=max
	return (data, norm)

def AddQuadraticTerms(data):
	(m,n) = data.shape
	for i in range(n):
		for j in range(n-i):
			data = np.c_[data,(data[:,i]*data[:,i+j]).reshape(m,1)]
	return data

def AddCubicTerms(data):
	(m,n) = data.shape
	for i in range(n):
		for j in range(n-i):
			#adding quadratic terms
			data = np.c_[data,(data[:,i]*data[:,i+j]).reshape(m,1)]
			for k in range(n-i-j):
				#adding cubic terms
				data = np.c_[data,(data[:,i]*data[:,i+j]*data[:,i+j+k]).reshape(m,1)]
	return data

def AddQuinticTerms(data):
	(m,n) = data.shape
	for i in range(n):
		for j in range(n-i):
			#adding quadratic terms
			data = np.c_[data,(data[:,i]*data[:,i+j]).reshape(m,1)]
			for k in range(n-i-j):
				#adding cubic terms
				data = np.c_[data,(data[:,i]*data[:,i+j]*data[:,i+j+k]).reshape(m,1)]
				for l in range(n-i-j-k):
					#adding quatric terms
					data = np.c_[data,(data[:,i]*data[:,i+j]*data[:,i+j+k]*data[:,i+j+k+l]).reshape(m,1)]
					for o in range(n-i-j-k-l):
						#adding quintic terms
						data = np.c_[data,(data[:,i]*data[:,i+j]*data[:,i+j+k]*data[:,i+j+k+l]*data[:,i+j+k+l+o]).reshape(m,1)]
	return data	

def AddSixticTerms(data):
	(m,n) = data.shape
	for i in range(n):
		for j in range(n-i):
			#adding quadratic terms
			data = np.c_[data,(data[:,i]*data[:,i+j]).reshape(m,1)]
			for k in range(n-i-j):
				#adding cubic terms
				data = np.c_[data,(data[:,i]*data[:,i+j]*data[:,i+j+k]).reshape(m,1)]
				for l in range(n-i-j-k):
					#adding quatric terms
					data = np.c_[data,(data[:,i]*data[:,i+j]*data[:,i+j+k]*data[:,i+j+k+l]).reshape(m,1)]
					for o in range(n-i-j-k-l):
						#adding quintic terms
						data = np.c_[data,(data[:,i]*data[:,i+j]*data[:,i+j+k]*data[:,i+j+k+l]*data[:,i+j+k+l+o]).reshape(m,1)]
						for p in range(n-i-j-k-l-o):
							#adding sixtic terms
							data = np.c_[data,(data[:,i]*data[:,i+j]*data[:,i+j+k]*data[:,i+j+k+l]*data[:,i+j+k+l+o]*data[:,i+j+k+l+o+p]).reshape(m,1)]
	return data

#get the data
trainori = pd.read_csv('train.csv')
testori = pd.read_csv('test.csv')
#merge the data and work with it (fill the gaps, extract some useful information)
testori["Survived"] = np.nan
combi = pd.concat([trainori,testori])
combi = TitanicExtractMeaning(combi)
#now that we are happy with our data, let's get back our training and testing sets 
set = combi.loc[np.logical_not(pd.isnull(combi.Age))]
train = set[0:800]
crossval = set[800:]
results = train.as_matrix(['Age'])
crossvalresults = crossval.as_matrix(['Age'])
del train['Age']
del crossval['Age']

traindata = np.array(train.as_matrix())
traindata = traindata.astype(float)
#add quadratic, cubic,... terms
print(traindata.shape)
traindata = AddCubicTerms(traindata)
print(traindata.shape)
#normalize the data
(traindata, norm) = normalize(traindata)


#adding the bias
(m,n) = traindata.shape
traindata = np.c_[np.ones(m),traindata]
#Now let's train our model
alpha = 0.1
lamb = 0
nbiter = 10000
drawhisto = True

weights = TrainLinearRegression(traindata, results, alpha, lamb, nbiter, drawhisto)
#now let's examine the performance of our model
#on our training set
prediction = PredictLinear(traindata, weights)
error = np.sum(np.square(results - prediction))/800/2
print('Error on training set:', error)


#on the validation set
crossvaldat = np.array(crossval.as_matrix())
#add quadratic terms
crossvaldat = AddCubicTerms(crossvaldat)
crossvaldat = crossvaldat/norm
#adding the bias
(m,n) = crossvaldat.shape
crossvaldat = np.c_[np.ones(m),crossvaldat]
prediction = PredictLinear(crossvaldat, weights)
print(prediction.shape)
error = np.sum(np.square(crossvalresults - prediction))/246/2
print('Error on validation set:', error)
print(set[800:820])
print(prediction[0:20])