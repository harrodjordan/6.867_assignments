
# Homework 2 for MIT 6.867 - Machine Learning 

import matplotlib.pyplot as plt
import pylab as pl
import numpy as np 

# Problem 3 

def getData(ifPlotData=True):
    # load the fitting data and (optionally) plot out for examination
    # return the X and Y as a tuple

    data = pl.loadtxt('p3/p3curvefitting.txt')

    X = data[0,:]
    Y = data[1,:]
    true_func = np.cos((np.pi*X)) + np.cos((2*np.pi*X))

    if ifPlotData:
        plt.plot(X,Y,'o')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    return X, Y

def basicPoly(X, Y, M, weights, sse=False):

	L = len(X)


	if sse == True:
		weights = np.zeros((L, M+1))
		#weights[0,:] = 1

	if M == 0:
		weights = np.ones((L, 1))

	for a in range(L):

		for b in range(M):

			temp = X[a]

			weights[a,b] = np.power(X[a],b)


	print(weights.shape)

	return np.sum(weights, axis=1) 

def SSEwithSGD(X, Y, M ,thresh = 0.01, alpha = 0.05, initial='zeros'):

	L = len(X)

	weights = np.zeros((L, M))

	new_weights = np.zeros(np.size(weights))

	if initial=='random':
		new_weights = np.random.rand(np.size(weights))

	temp = 10


	while temp > thresh:

		weights = basicPoly(X, Y, M, weights)

		optimal_func = np.transpose(weights)*X 

		for a in range(L):

			

			new_weights[a] = (Y[a] - optimal_func[a])*np.power(X[a],M)

			temp = (Y[a] - optimal_func[a])*np.power(X[a],M)

		weights = weights - alpha*new_weights

	return weights 

def SSEwithBGD(X, Y, M ,thresh = 0.01, alpha = 0.05, initial='zeros'):

	L = len(X)

	weights = np.zeros((L, M))

	new_weights = np.zeros(np.size(weights))

	if initial=='random':
		new_weights = np.random.rand(np.size(weights))

	while temp > thresh:

		weights = basicPoly(X, Y, M)

		optimal_func = X*weights 

		new_weights[a,b] = (Y - optimal_func)*np.power(X,b)

		weights = weights - alpha*new_weights

	return weights 


def cosinePoly(X, Y, M):

	L = len(X)

	weights = np.zeros((L, M))

	if M == 0:
		weights = np.ones((L, 1))
		
	for a in range(L):

		for b in range(M):

			temp = X[a]

			weights[a,b] = np.cos(X[a]*b)

	print(weights.shape)


	return np.sum(weights, axis=1)



def showData(weights, M):
    # load the fitting data and (optionally) plot out for examination
    # return the X and Y as a tuple

    data = pl.loadtxt('p3/p3curvefitting.txt')

    X = data[0,:]
    Y = data[1,:]
    true_func = np.cos((np.pi*X)) + np.cos((2*np.pi*X))
    optimal_func = np.transpose(weights)*X
    #optimal_func = np.poly1d(np.polyfit(X, Y, M+1))
    print(optimal_func.shape)

    plt.plot(X,Y,'o')
    plt.plot(X, true_func, color="yellow")
    plt.plot(X, np.transpose(optimal_func), color="red")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cosine Poly Regression M = {0}\n.png'.format(M))
    plt.show()
    plt.savefig('Cosine Poly Regression M = {0}\n.png'.format(M))
    plt.close()

    #add something to save figure 

print("Importing data.....")

[X, Y] = getData();

print("Data Import Complete")

# for i in range(11):

# 	print('Linear Regression M = {0}\n'.format(i))

# 	weights = basicPoly(X, Y, i, 0, sse=True)
# 	showData(weights, i)


print('Training SSE Poly Model with SGD and M = 10.png')
weights = SSEwithSGD(X, Y, 10, 0.001, 1e-3, 'zeros')
showData(weights, M)

# for (alpha, thresh, initial, M) in range(1, 1, 1, 10):
# 	print("Training SSE Poly Model Batch and M=" + i)
# 	weights = SSEwithGD(X, Y, M, thresh, alpha, initial, grad='batch')
# 	showData(weights, M)

# for i in range(8):
# 	print('Training Cosine Poly Model M = {0}\n.png'.format(i))
# 	weights = cosinePoly(X, Y, i+1)
# 	showData(weights, i+1)



        

# Problem 4 - See folder p4 with most of the code given as part of the assignment  
