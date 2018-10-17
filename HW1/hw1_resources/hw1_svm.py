import numpy as np
import cvxopt 
from cvxopt import matrix, solvers
import scipy.signal
import sklearn.metrics.pairwise as pair 


def SVM(vectors, labels, C):



	print(vectors.shape)
	print(labels.shape)

	shape_l, shape_w = vectors.shape 

	# A = np.identity(shape_l)

	# alpha = np.vstack((-1*A, A))

	# print(alpha.shape)

	# C_v = np.zeros((shape_l,1)).fill(C)

	# C_matrix = np.vstack((np.zeros((shape_l, 1)), C_v))

	# C_matrix = C_matrix.astype(np.double)


	I_d = np.identity(shape_l)
	G = np.vstack((-1*I_d, I_d))

	C_col = np.zeros((shape_l,1))
	C_col.fill(C)
	h = np.vstack( (np.zeros((shape_l,1)), C_col) )

	q  = np.zeros((shape_l,1))
	q.fill(-1)


	bias = 0.0

	P = matrix(np.outer(labels,labels))

	q = matrix(q)

	G = matrix(G)

	h = matrix(h)

	A = matrix(labels.T)

	b = matrix(bias)

	solve = solvers.qp(P, q	, G, h, A, b)
	alpha = np.array(solve['x'])

	s_vec = np.where((alpha > 1e-4) & (alpha < C))

	s_vec = s_vec[0]

	new_alpha = alpha[s_vec].reshape(-1, 1)
	new_vec_X = vectors[s_vec.flatten()].reshape(-1, shape_w)
	new_vex_y = labels[s_vec].reshape(-1, 1)


	return alpha, new_alpha, new_vec_X, new_vex_y, s_vec


def two_dimen_test(C=1):

	return None


def linear_gram(X, m=1):
	'''Given a dataset, computes the gram matrix using k(x,y) = (1+xy)^m, m=1 '''
	
	def k(x,y):

		try:
			mf =  (1+np.dot(x,y))**m
		except ValueError:
			print("valuerr", x, y)
			return None
		return mf        
	return pair.linear_kernel(X) #compute_gram(X,k)

def gaussian_gram(x, gamma):
	'''Given a dataset and bandwidth sigma, computes the Gaussian RBF kernel matrix'''

	# recast x matrix as floats
	x = np.asfarray(x)

	# get a matrix where the (i, j)th element is |x[i] - x[j]|^2
	pt_sq_norms = (x ** 2).sum(axis=1)
	dists_sq = np.dot(x, x.T)
	dists_sq *= -2.0
	dists_sq += pt_sq_norms.reshape(-1, 1)
	dists_sq += pt_sq_norms
	# turn into an RBF gram matrix
	km = dists_sq; del dists_sq
	km *= -1.*gamma
	K = np.exp(km)  # exponentiates in-place
	return K 

def compute_gram(X, k):
	''' Given a function k and a datasdet X, computes the Gram matrix
	slow as fudge'''
	K = np.ones((400,400))

	K = scipy.signal.convolve2d(X, k, mode='same')
	return K


def SVM_with_kernel(vectors, labels, C, kernel):


	print(vectors.shape)
	print(labels.shape)

	K = compute_gram(vectors,kernel)

	shape_l, shape_w = vectors.shape 

	I_d = np.identity(shape_l)
	G = np.vstack((-1*I_d, I_d))

	C_col = np.zeros((shape_l,1))
	C_col.fill(C)
	h = np.vstack( (np.zeros((shape_l,1)), C_col) )

	q  = np.zeros((shape_l,1))
	q.fill(-1)


	bias = 0.0

	P = matrix(np.outer(labels,labels)*kernel)

	q = matrix(q)

	G = matrix(G)

	h = matrix(h)

	A = matrix(labels.T)

	b = matrix(bias)

	solve = solvers.qp(P, q	, G, h, A, b)
	alpha = np.array(solve['x'])

	s_vec = np.where((alpha > 1e-4) & (alpha < C))

	s_vec = s_vec[0]

	new_alpha = alpha[s_vec].reshape(-1, 1)
	new_vec_X = vectors[s_vec.flatten()].reshape(-1, shape_w)
	new_vex_y = labels[s_vec].reshape(-1, 1)


	return alpha, new_alpha, new_vec_X, new_vex_y, s_vec



#PART A: PEGASOS 
'''
The following pseudo-code is a slight variation on the Pegasos learning 
algorithm, with a ﬁxed iteration count and non-random presentation of the 
training data. Implement it, and then add a bias term (w0) to the hypothesis, 
but take care not to penalize the magnitude of w0. Your function should output
classiﬁer weights for a linear decision boundary.
'''

def train_linearSVM(X, Y, L, max_epochs):
	debug = False 
	t = 0
	assert L != 0; 'Lambda must be non-zero'
	# Initialise w
	if debug: 
		print 'X data', X
		print 'Y data', Y

	n. d = X.shape # n = number of data points, d = dimension
	nY, dY = Y.shape # n = number of points, c should be 1

	# TO DO: ADD BIAS TERM; DO NOT PENALIZE IT

	assert n == nY; 'X and Y should have same number of sammples'
	w = np.empty([d, 1])	# w is the weight vector of dimension d x 1

	if debug: print 'weight matrix: ',w

	epoch = 0
	while epoch < max_epochs:
		for i in range(n):
			t += 1
			eta = 1.0/(t*L)
			if Y[i] * np.dot(w, X[i]) < 1:
				w = (1-eta*L)*w + eta*Y[i]*X[i]
			else:
				w = (1-eta*L)*w 
		epoch += 1

	return w


#PART B
'''
Test various values of the regularization constant, L = 2 , . . . , 2e−10 . Observe the the margin 
(distance between the decision boundary and margin boundary) as a function of L. 
Does this match your understanding of the objective function?
'''

L_test = [2e-(i+1) for i in range(10)]


#PART C: KERNELIZED SOFT SVM
'''
We can also solve the following kernelized Soft-SVM problem with a few extensions to the above algorithm. 
Rather than maintaining the w vector in the dimensionality of the data, we maintain α coefficients for 
each training instance.
Implement a kernelized version of the Pegasos algorithm. It should take in a Gram matrix, where entry 
i, j is K(x(i), x(j)) = phi(x(i)) * phi(x(j)), and should should output the support vector values, alpha,
or a function that makes a prediction for a new input. In this version, you do not need to add a bias term.
'''

def train_gaussianSVM(X, Y, K, L, max_epochs):
	debug = False
	t = 0
	assert L != 0; 'Lambda must be non-zero'
	# Initialise w
	if debug: 
		print 'X data', X
		print 'Y data', Y

	n. d = X.shape # n = number of data points, d = dimension
	nY, dY = Y.shape # n = number of points, c should be 1

	# TO DO: ADD BIAS TERM; DO NOT PENALIZE IT

	assert n == nY; 'X and Y should have same number of sammples'
	alpha = np.empty([d, 1])	# a is the vector of alphas of dimension d x 1

	if debug: print 'Initial alpha matrix: ',a

	epoch = 0
	while epoch < max_epochs:
		for i in range(n):
			t += 1
			eta = 1.0/(t*L)

			# Compute Y[i] * sum_j a_j K(x_j, x_i)

			#grab relevant column of K
			K_col = K[:,i]
			S = np.dot(alpha.T, K_col)
			if Y[i] * S < 1:
				alpha[i] = (1-eta*L)*alpha[i] + eta*Y[i]
			else:
				alpha[i] = (1-eta*L)*alpha[i]
		epoch += 1

	return alpha








