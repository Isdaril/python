import numpy as np
import matplotlib.pyplot as plt
import math
import os

def sigmoid(x):
	x = x.astype(float)
	sig = 1/(1 + np.exp(-x))
	return sig

def MultTerms(data, index):
	#technical function used by Loop to perform its duty
	#data : the data you want to add terms to
	#the indexes of the columns you want to multiply
	(m,k) = data.shape
	result = np.ones((m,1))
	n = len(index)
	for i in range(n):
		result = (result*data[:,index[i]].reshape(m,1))
	return result

def Loop(data, result, index, n, i, nt):
	#technical function used by AddTerms to do its job
	#yes i'm aware this is probably poorly implemented, the complexity is probably along O(!nt)
	#data : the initial data you want to add terms to (this needs to be stored)
	#result : the data you will ultimately returns
	#index : an intermediate list used to store the list of index we have to multiply
	#n : the length of the current loop
	#i : the position we are in within that loop
	#nt : the level we have to calculate within that loop
	if nt == 1:
		for j in range(n-i):
			index.append(i+j)
			result = np.c_[result, MultTerms(data,index)]
			del index[-1]
	else:
		for j in range(n-i):
			index.append(i+j)
			result = np.c_[result, MultTerms(data,index)]
			result = Loop(data, result, index, n, i+j, nt-1)
			del index[-1]
	return result

def AddTerms(data, level):
	#This function add quadratic, cubic... terms to the data
	#data : the data you want to add terms to
	#level : the level of terms we want to add (2 = quad, 3 = cubic and so on...)
	(m,n) = data.shape
	result = np.empty((m,0))
	index = []
	return Loop(data, result, index, n, 0, level)

def TrainLinearRegression(data, results, alpha, lamb, nbiter, drawhisto):
	#data is the normalized training set with bias added
	#results is the vector of results (containing 0 or 1)
	#alpha is the learning rate
	#lambda is the regularization constant
	#nbiter is the number of step we take to train our logisic regression
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

def TrainLogisticRegression(data, results, alpha, lamb, nbiter, drawhisto):
	#data is the normalized training set with bias added
	#results is the vector of results (containing 0 or 1)
	#alpha is the learning rate
	#lambda is the regularization constant
	#nbiter is the number of step we take to train our logistic regression
	(m,n) = data.shape
	#normalized random initialization
	weights = np.random.random(n) - 0.5
	weights = weights.reshape(n,1)
	#create a vector of costs
	costs = []
	for i in range(nbiter):
		#computing the regularized gradient (we exclude the bias term from the regularized expression)
		sig = sigmoid(np.dot(data,weights))
		grad = 1/m * np.dot(np.transpose(data), sig - results) + lamb/m*np.insert(weights[1:,:],0, 0).reshape(n,1)
		weights = weights - alpha*grad
		if drawhisto:
			#compute cost
			cost = np.sum(np.square(predictLogistic(data, weights)-results))/m
			costs.append(cost)
	return (weights, costs)

def predictLogistic(data, weights):
	predict = np.rint(sigmoid(np.dot(data,weights)))
	return predict

def normalize(data):
	(m,n) = data.shape
	norm = np.zeros(n)
	for i in range(n):
		maxi = np.amax(data[:,i])
		data[:,i] = data[:,i]/maxi
		norm[i]=maxi
	return (data, norm)

def getASlice(data, results, n, m):
	#this function cuts the data into two pieces at place n for m terms
	crossVal = data[n:n+m,:]
	crossValresults = results[n:n+m,:]
	train = np.concatenate((data[0:n,:],data[n+m:,:]), axis=0)
	trainresults = np.concatenate((results[0:n,:],results[n+m:,:]), axis=0)
	return (train, trainresults, crossVal, crossValresults)

def RandomSplicingTest(data, results, n, N, alpha, lamb, drawhisto, level):
	#get the data and slice it in n different training and cross validation sets
	#then, perform a training and testing on all those sets
	print('starting training multiple sets')
	m = len(data)
	crossSize = math.floor(m/3)
	rand = np.random.randint(m - crossSize, size=n)
	data = AddTerms(data,level)
	(data, norm) = normalize(data)
	errors = []
	for i in range(n):
		(train, trainresults, crossVal, crossValresults) = getASlice(data, results, rand[i], crossSize)
		print('training set ',i,' in progress...')
		(weights, costs) = TrainLogisticRegression(train, trainresults, alpha, lamb, N, drawhisto)
		prediction = predictLogistic(train, weights)
		error = np.sum(np.square(trainresults - prediction))/(m - crossSize)
		print('error on training set is :', error)
		prediction = predictLogistic(crossVal, weights)
		error = np.sum(np.square(crossValresults - prediction))/crossSize
		print('error on CV set is :', error)
		errors.append(error)
	mean = sum(errors)/n
	worse = max(errors)
	best = min(errors)
	print('best error on CV is :', best)
	print('mean error on CV set is :', mean)
	print('worse error on CV is :', worse)

def RandomSplicingNN(data, results, N, model, batchSize, epochs,drawhisto):
	(m,n) = data.shape
	crossSize = math.floor(m/3)
	rand = np.random.randint(m - crossSize, size=N)
	(data, norm) = normalize(data)
	#data = data.reshape(1,m,n)
	#results = results.reshape(1,m,1)
	errors = []
	wSave = model.get_weights()
	for i in range(N):
		#reinitializes the weights
		model.set_weights(wSave)

		(train, trainresults, crossVal, crossValresults) = getASlice(data, results, rand[i], crossSize)

		if drawhisto:
			crossError = []
			trainError = []
			for i in range(epochs):
				print('epoch : ',i,'/',epochs)
				model.fit(train, trainresults, batch_size=batchSize, epochs=1, verbose=0, validation_data=(crossVal,crossValresults), shuffle=True)
				trainEval = model.evaluate(train, trainresults, batch_size=batchSize, verbose=0)[1]
				crossEval = model.evaluate(crossVal, crossValresults, batch_size=batchSize, verbose=0)[1]
				trainError.append(trainEval)
				crossError.append(crossEval)
			trainLine, = plt.plot(trainError, label='training accuracy')
			crossLine, = plt.plot(crossError, label='CV accuracy')
			plt.legend(handles=[trainLine,crossLine])
			plt.xlabel('Iterations')
			plt.ylabel('Accuracy')
			plt.show()
		else:
			model.fit(train, trainresults, batch_size=batchSize, epochs=epochs, verbose=1, validation_data=(crossVal,crossValresults), shuffle=True)
		evaluate = model.evaluate(crossVal, crossValresults, batch_size=batchSize, verbose=0)
		error = evaluate[1]
		print('Accuracy on CV set is :', error)
		errors.append(error)
	mean = sum(errors)/N
	best = max(errors)
	worse = min(errors)
	print('best accuracy on CV is :', best)
	print('mean accuracy on CV set is :', mean)
	print('worse accuracy on CV is :', worse)


def PlotDifferentAlphas(data, results, alphaSet, n, lamb, level):
	data = AddTerms(data,level)
	(data, norm) = normalize(data)
	lines = []
	for i in range(len(alphaSet)):
		print('entering reg ',i)
		(weights, costs) = TrainLogisticRegression(data, results, alphaSet[i], lamb, n, True)
		alphalabel = 'alpha = ' + str(alphaSet[i])
		alphaline, = plt.plot(costs[10:], label=alphalabel)
		lines.append(alphaline)
	plt.legend(handles=lines)
	plt.xlabel('Iterations')
	plt.ylabel('Cost')
	plt.show()

def PlotDifferentLambdas(data, results, lambdaSet, alpha, n, N, level):
	data = AddTerms(data,level)
	(data, norm) = normalize(data)
	k = len(lambdaSet)
	m = len(data)
	crossSize = math.floor(m/3)
	rand = np.random.randint(m - crossSize, size=N)
	alphas = np.empty([3,k])
	for i in range(k):
		errors = []
		print('alpha ',i)
		for j in range(N):
			print('entering reg ',j)
			(train, trainresults, crossVal, crossValresults) = getASlice(data, results, rand[j], crossSize)
			(weights, costs) = TrainLogisticRegression(train, trainresults, alpha, lambdaSet[i], n, False)
			prediction = predictLogistic(crossVal, weights)
			error = np.sum(np.square(crossValresults - prediction))/crossSize
			errors.append(error)
		alphas[0,i] = min(errors)
		alphas[1,i] = sum(errors)/N
		alphas[2,i] = max(errors)
	bestLine, = plt.plot(lambdaSet, alphas[0,:],label='best')
	meanLine, = plt.plot(lambdaSet, alphas[1,:],label='mean')
	worseLine, = plt.plot(lambdaSet, alphas[2,:],label='worse')
	plt.legend(handles=[bestLine,meanLine,worseLine])
	plt.xlabel('Lambda')
	plt.ylabel('Cost')
	plt.grid(True)
	plt.show()
