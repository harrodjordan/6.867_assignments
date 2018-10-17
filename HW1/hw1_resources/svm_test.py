from numpy import *
from plotBoundary import *
import pylab as pl
import numpy as np
# import your SVM training code
import hw1_svm


# parameters
name = '4'
print ('======Training======')
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()

# Carry out training, primal and/or dual
### TODO ###

# X = np.array([[2,2],[2,3], [0,-1], [-3,-2]])
# Y = np.array([1, 1, -1, -1])
# Y = np.reshape(Y, (4,1))

C = 1
K = hw1_svm.gaussian_gram(X, gamma = 1)
K = hw1_svm.linear_gram(X)

alpha, SVM_alpha, SVM_X, SVM_Y, support = hw1_svm.SVM_with_kernel(X, Y, C, K)

# Define the predictSVM(x) function, which uses trained parameters
### TODO ###

def column_kernel(SVM_X,x):
	'''
	Given an array of X values and a new x to predict, 
	computes the  vector whose i^th entry is k(SVM_X[i],x)
	'''
	def k(y):
		#return np.dot(x,y) # returns the identity kernel
	
		#return (1+np.dot(x,y)) # returns the linear basis kernel
		
		gamma = 2
		sqr_diff = np.linalg.norm(x-y)**2
		sqr_diff *= -1.0*gamma 
		return np.exp(sqr_diff)
		
	return np.apply_along_axis(k, 1, SVM_X ).reshape(-1,1)

def get_prediction_constants():

	#print("Prediction Constants on SVM")
	ay = SVM_alpha*SVM_Y
	# get gram matrix for only support X values
	SVM_K = K[support]
	SVM_K = SVM_K.T[support]
	SVM_K = SVM_K.T

	# compute bias
	bias = np.nansum([SVM_Y[i] - np.dot(ay.T, SVM_K[i]) for i in range(len(SVM_Y))]) / len(SVM_Y)

	return ay, bias


def predictSVM(x):

	#print("Predicting on SVM")

	ay, bias = get_prediction_constants()

	x = x.reshape(1, -1)
	
	# predict new Y output
	kernel = column_kernel(SVM_X, x)
	y = np.dot(ay.T, kernel)

	return y + bias

def classification_error(X_train, Y_train):
	''' Computes the error of the classifier on some training set'''
	n,d = X_train.shape
	incorrect = 0.0
	for i in range(n):
		if predictSVM(X_train[i]) * Y_train[i] < 0:
			incorrect += 1
	return incorrect/n
	
train_err = classification_error(X, Y)



# plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')
pl.savefig('SVMtrain_'+str(name)+'.png')


print ('======Validation======')
# load data from csv files
validate = loadtxt('data/data'+name+'_validate.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]
# plot validation results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')
pl.show()
pl.savefig('SVMvalidate_'+str(name)+'.png')

validation_err = classification_error(X, Y)


f = open('errors for dataset '+str(name)+'.txt', 'w')
f.write('Train err: ')
f.write(str(train_err))
f.write('\n')
f.write('Validate err: ')
f.write(str(validation_err))
f.write('\n')
f.write('Number of SVMs: ')
f.write(str(len(SVM_Y)))
f.close()

print ('Done plotting...')
