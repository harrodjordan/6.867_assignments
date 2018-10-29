import numpy as np
import math
import gradient_check as gc
import loadFittingDataP3 as lf
import matplotlib.pyplot as plt
import itertools
import pylab
import scipy
from sklearn import linear_model


def fPhi(w):
	(X,y) = lf.getData(False)
	N = w.shape[0]
	Phi = np.zeros((X.shape[0],N))
	f = np.zeros((X.shape[0]))
	for ii in range(X.shape[0]):
		Phi[ii,:]=[X[ii]**i for i in range(N)]

	return np.linalg.norm(Phi.dot(w)-y)

def fPhi_i(w,i):
	(X,y) = lf.getData(False)
	N = w.shape[0]
	Phi = np.zeros((N))
	Phi = np.array([X[i]**ii for ii in range(N)])

	return (Phi.dot(w)-y[i])**2

def dfPhi(w):
	return gc.eval_numerical_gradient(fPhi,w)

def fPhicos(w):
	(X,y) = lf.getData(False)
	N = w.shape[0]
	Phi = np.zeros((X.shape[0],N))
	f = np.zeros((X.shape[0]))
	for ii in range(X.shape[0]):
		Phi[ii,:]=[math.cos(X[ii]*np.pi*i) for i in range(1,N+1)]

	return np.linalg.norm(Phi.dot(w)-y)**2

def fPhicos_i(w,i):
	(X,y) = lf.getData(False)
	N = w.shape[0]
	Phi = np.zeros((N))
	Phi = np.array([math.cos(X[ii]*np.pi*i) for i in range(1,N+1)])

	return (Phi.dot(w)-y[i])**2

def sdg(f,xold, lr=0.001, tol=1e-12):

	t = 0.0
	(X,y) = lf.getData(False)
	#i = np.random.randint(X.shape[0]-1)
	theta_exact = np.linalg.inv(Phi.T.dot(Phi)).dot(Phi.T.dot(y))
	normlist = []
	i = 1
	lrold = lr
	conv = np.sum([gc.eval_numerical_gradient_i(f, xold, ii, verbose=False) for ii in range(X.shape[0]-1)])
	while (conv > tol):
		xnew = xold - lr*gc.eval_numerical_gradient_i(f, xold, i,verbose=False)
		t += 1.0
		lr = lrold/t
		#print(np.linalg.norm(xold-xnew))
		normlist.append(abs(fPhi(theta_exact)-fPhi(xnew)))
		if i%(X.shape[0]-1)==0:
			i = 0
		else:
			i += 1
		#if t%20==0:
			#print(xold)
			#conv = np.sum([gc.eval_numerical_gradient_i(f, xold, ii, verbose=False) for ii in range(X.shape[0]-1)])
			#conv = J(xold)
			#print(conv)
			#theta = exact_batch_min() 
		if (np.linalg.norm(xold-xnew) < 10.0e-7):
			print(xold)
			#conv = J(xold)
			conv = np.sum([gc.eval_numerical_gradient_i(f, xold, ii, verbose=False) for ii in range(X.shape[0]-1)])
			print(conv)
			normlist.append(conv)
			return xold, normlist
		if t > 200:
			return xold, normlist
		xold = xnew
		
	#conv = np.sum([gc.eval_numerical_gradient_i(f, xold, ii, verbose=False) for ii in range(X.shape[0]-1)])
	#normlist.append(conv)
	print(normlist)
	print(xold)
	return xold, normlist

#####################################################################################
(X,y) = lf.getData(False)
xs = np.linspace(0,1,40)
# for N in range (1,11,2):

# 	Phi = np.zeros((X.shape[0],N))
# 	for ii in range(X.shape[0]):
# 		#cosine
# 		#Phi[ii,:]=[math.cos(X[ii]*np.pi*i) for i in range(1,N+1)]
# 		#polynomial
# 		Phi[ii,:]=[X[ii]**i for i in range(N)]
# 		#for s in itertools.combinations(X,9):
# 		#	print(s)

# 	#ml of LS
# 	w = np.linalg.inv(Phi.T.dot(Phi)).dot(Phi.T.dot(y))
# 	#regularized LS
# 	l = 0.1
# 	w_reg = np.linalg.inv(l*np.eye(Phi.shape[1])+Phi.T.dot(Phi)).dot(Phi.T.dot(y))

# 	w0 = w +0.1*np.random.randn(w.shape[0])
# 	wgc_batch = gc.numeric_grad_descent(fPhi,w0,lr=0.02, tol=5e-4)
# 	wgc_sdg, normlist = sdg(fPhi_i,w0)

# 	#print(gc.eval_numerical_gradient_i(fPhi_i,w0,1))

# 	f = np.zeros((X.shape[0]))
# 	f_sdg = np.zeros((X.shape[0]))
# 	for ii in range(X.shape[0]):
# 		#polynomial
# 		#f[ii] = w.T.dot(Phi[ii])
# 		#cosine
# 		#f[ii] = w.dot([math.cos(X[ii]*np.pi*i) for i in range(1,N+1)])
# 		#regularized
# 		f[ii] = wgc_batch.dot(Phi[ii])
# 		f_sdg[ii] = wgc_sdg.dot(Phi[ii])

# 	pl1 = plt.plot(X,y,'o',label='Given Data')
# 	pl2 = plt.plot(X,f,'s',label='Batch')
# 	pl3 = plt.plot(X,f_sdg,'1',label='SGD')
# 	plt.xlabel('x')
# 	plt.ylabel('y')
# 	#plt.show()
# 	# #pylab.legend(handles=[pl1,pl2])
# 	plt.legend(loc='upper right')
# 	plt.savefig('../../figures/ps2_check_batchsdg_'+str(N)+'.pdf')
# 	plt.clf()

# for N in range (1,11,1):

# 	Phi = np.zeros((X.shape[0],N))
# 	for ii in range(X.shape[0]):
# 		#cosine
# 		#Phi[ii,:]=[math.cos(X[ii]*np.pi*i) for i in range(1,N+1)]
# 		#polynomial
# 		Phi[ii,:]=[X[ii]**i for i in range(N)]
# 		#for s in itertools.combinations(X,9):
# 		#	print(s)

# 	#ml of LS
# 	w = np.linalg.inv(Phi.T.dot(Phi)).dot(Phi.T.dot(y))
# 	#regularized LS
# 	l = 0.1
# 	w_reg = np.linalg.inv(l*np.eye(Phi.shape[1])+Phi.T.dot(Phi)).dot(Phi.T.dot(y))

# 	w0 = w +0.1*np.random.randn(w.shape[0])
# 	wgc_batch = gc.numeric_grad_descent(fPhi,w0,lr=0.02, tol=5e-4)
# 	wgc_sdg, normlist = sdg(fPhi_i,w0)
# 	wgc_batch_c = gc.numeric_grad_descent(fPhicos,w0,lr=0.01, tol=1e-4)
	
# 	Phi = np.zeros((X.shape[0],N))
# 	for ii in range(X.shape[0]):
# 		#cosine
# 		Phi[ii,:]=[math.cos(X[ii]*np.pi*i) for i in range(1,N+1)]
# 		#polynomial
# 		#Phi[ii,:]=[X[ii]**i for i in range(N)]

# 	w = np.linalg.inv(Phi.T.dot(Phi)).dot(Phi.T.dot(y))
# 	w0 = w +0.1*np.random.randn(w.shape[0])
# 	wgc_sdg_c, normlist = sdg(fPhicos_i,w0)

# 	#print(gc.eval_numerical_gradient_i(fPhi_i,w0,1))

# 	# f = np.zeros((X.shape[0]))
# 	# f_sdg = np.zeros((X.shape[0]))
# 	# for ii in range(X.shape[0]):
# 	# 	#polynomial
# 	# 	#f[ii] = w.T.dot(Phi[ii])
# 	# 	#cosine
# 	# 	#f[ii] = w.dot([math.cos(X[ii]*np.pi*i) for i in range(1,N+1)])
# 	# 	#regularized
# 	# 	f[ii] = wgc_batch.dot(Phi[ii])
# 	# 	f_sdg[ii] = wgc_sdg.dot(Phi[ii])

# 		#f = np.zeros((X.shape[0]))
# 	#f_sdg = np.zeros((X.shape[0]))
# 	fp_b = np.zeros((xs.shape[0]))
# 	fp_sdg = np.zeros((xs.shape[0]))
# 	fc_b = np.zeros((xs.shape[0]))
# 	fc_sdg = np.zeros((xs.shape[0]))
# 	Phip = np.zeros((xs.shape[0],N))
# 	Phic = np.zeros((xs.shape[0],N))
# 	for ii in range(xs.shape[0]):
# 		Phip[ii,:]=[xs[ii]**i for i in range(N)]
# 		Phic[ii,:]=[math.cos(xs[ii]*np.pi*i) for i in range(1,N+1)]
# 		#polynomial
# 		#f[ii] = w.T.dot(Phi[ii])
# 		#cosine
# 		#f[ii] = w.dot([math.cos(X[ii]*np.pi*i) for i in range(1,N+1)])
# 		#regularized
# 		fp_b[ii] = wgc_batch.dot(Phip[ii])
# 		fp_sdg[ii] = wgc_sdg.dot(Phip[ii])
# 		fc_b[ii] = wgc_batch_c.dot(Phic[ii])
# 		fc_sdg[ii] = wgc_sdg_c.dot(Phic[ii])

# 	pl1 = plt.plot(X,y,'o',label='Given Data')
# 	pl2 = plt.plot(xs,fp_b,label='Polynomial Batch')
# 	pl3 = plt.plot(xs,fp_sdg,label='Polynomial SGD')
# 	pl4 = plt.plot(xs,fc_b,label='Cos Batch')
# 	pl5 = plt.plot(xs,fc_sdg,label='Cos SGD')

# 	# pl1 = plt.plot(X,y,'o',label='Given Data')
# 	# pl2 = plt.plot(X,f,'s',label='Batch')
# 	# pl3 = plt.plot(X,f_sdg,'1',label='SGD')
# 	plt.xlabel('x')
# 	plt.ylabel('y')
# 	#plt.show()
# 	# #pylab.legend(handles=[pl1,pl2])
# 	plt.legend(loc='upper right')
# 	plt.savefig('../../figures/ps2_check_batchsdg_'+str(N)+'.pdf')
# 	plt.clf()

#For Cos basis
# for N in range (1,11):

# 	Phi = np.zeros((X.shape[0],N))
# 	for ii in range(X.shape[0]):
# 		#cosine
# 		Phi[ii,:]=[math.cos(X[ii]*np.pi*i) for i in range(1,N+1)]
# 		#polynomial
# 		#Phi[ii,:]=[X[ii]**i for i in range(N)]
# 		#for s in itertools.combinations(X,9):
# 		#	print(s)

# 	#ml of LS
# 	w = np.linalg.inv(Phi.T.dot(Phi)).dot(Phi.T.dot(y))
# 	#regularized LS
# 	l = 0.1
# 	w_reg = np.linalg.inv(l*np.eye(Phi.shape[1])+Phi.T.dot(Phi)).dot(Phi.T.dot(y))

# 	w0 = w +0.1*np.random.randn(w.shape[0])
# 	wgc_batch = gc.numeric_grad_descent(fPhicos,w0,lr=0.01, tol=1e-4)
# 	wgc_sdg, normlist = sdg(fPhicos_i,w0)

# 	#print(gc.eval_numerical_gradient_i(fPhi_i,w0,1))

# 	f = np.zeros((X.shape[0]))
# 	f_sdg = np.zeros((X.shape[0]))
# 	for ii in range(X.shape[0]):
# 		#polynomial
# 		#f[ii] = w.T.dot(Phi[ii])
# 		#cosine
# 		#f[ii] = w.dot([math.cos(X[ii]*np.pi*i) for i in range(1,N+1)])
# 		#regularized
# 		f[ii] = wgc_batch.dot(Phi[ii])
# 		f_sdg[ii] = wgc_sdg.dot(Phi[ii])

# 	pl1 = plt.plot(X,y,'o',label='Given Data')
# 	pl2 = plt.plot(X,f,'s',label='Batch')
# 	pl3 = plt.plot(X,f_sdg,'1',label='SGD')
# 	plt.xlabel('x')
# 	plt.ylabel('y')
# 	#plt.show()
# 	# #pylab.legend(handles=[pl1,pl2])
# 	plt.legend(loc='upper right')
# 	plt.savefig('../../figures/ps2_batchsdg_cos'+str(N)+'.pdf')
# 	plt.clf()

#For Poly and Cos basis
# for N in range (2,12,2):

# 	Phi = np.zeros((X.shape[0],N))
# 	for ii in range(X.shape[0]):
# 		#cosine
# 		Phi[ii,:]=[math.cos(X[ii]*np.pi*i) for i in range(1,N+1)]
# 		#polynomial
# 		#Phi[ii,:]=[X[ii]**i for i in range(N)]
# 		#for s in itertools.combinations(X,9):
# 		#	print(s)

# 	#ml of LS
# 	w = np.linalg.inv(Phi.T.dot(Phi)).dot(Phi.T.dot(y))
# 	#regularized LS
# 	l = 0.1
# 	w_reg = np.linalg.inv(l*np.eye(Phi.shape[1])+Phi.T.dot(Phi)).dot(Phi.T.dot(y))

# 	w0 = w +0.1*np.random.randn(w.shape[0])
#  	wgc_batch_p = gc.numeric_grad_descent(fPhi,w0,lr=0.01, tol=5e-4)
#  	wgc_sdg_p, normlist = sdg(fPhi_i,w0)
# 	wgc_batch = gc.numeric_grad_descent(fPhicos,w0,lr=0.01, tol=1e-4)
# 	wgc_sdg, normlist = sdg(fPhicos_i,w0)

# 	#print(gc.eval_numerical_gradient_i(fPhi_i,w0,1))

# 	#f = np.zeros((X.shape[0]))
# 	#f_sdg = np.zeros((X.shape[0]))
# 	fp_b = np.zeros((xs.shape[0]))
# 	fp_sdg = np.zeros((xs.shape[0]))
# 	fc_b = np.zeros((xs.shape[0]))
# 	fc_sdg = np.zeros((xs.shape[0]))
# 	Phip = np.zeros((xs.shape[0],N))
# 	Phic = np.zeros((xs.shape[0],N))
# 	for ii in range(xs.shape[0]):
# 		Phip[ii,:]=[xs[ii]**i for i in range(N)]
# 		Phic[ii,:]=[math.cos(xs[ii]*np.pi*i) for i in range(1,N+1)]
# 		#polynomial
# 		#f[ii] = w.T.dot(Phi[ii])
# 		#cosine
# 		#f[ii] = w.dot([math.cos(X[ii]*np.pi*i) for i in range(1,N+1)])
# 		#regularized
# 		fp_b[ii] = wgc_batch_p.dot(Phip[ii])
# 		fp_sdg[ii] = wgc_sdg_p.dot(Phip[ii])
# 		fc_b[ii] = wgc_batch.dot(Phic[ii])
# 		fc_sdg[ii] = wgc_sdg.dot(Phic[ii])

# 	pl1 = plt.plot(X,y,'o',label='Given Data')
# 	pl2 = plt.plot(xs,fp_b,label='Polynomial Batch')
# 	pl3 = plt.plot(xs,fp_sdg,label='Polynomial SGD')
# 	pl4 = plt.plot(xs,fc_b,label='Cos Batch')
# 	pl5 = plt.plot(xs,fc_sdg,label='Cos SGD')
# 	plt.xlabel('x')
# 	plt.ylabel('y')
# 	#plt.show()
# 	# #pylab.legend(handles=[pl1,pl2])
# 	plt.legend(numpoints=1,loc='upper right',fontsize = 'x-small')
# 	plt.savefig('../../figures/ps2_batchsdg_polycos'+str(N)+'.pdf')
# 	plt.clf()

#for N in range (1,11,2):
# for N in range (1,12,2):
# 	Phi = np.zeros((X.shape[0],N))
# 	for ii in range(X.shape[0]):
# 		#cosine
# 		#Phi[ii,:]=[math.cos(X[ii]*np.pi*i) for i in range(1,N+1)]
# 		#polynomial
# 		Phi[ii,:]=[X[ii]**i for i in range(N)]
# 		#for s in itertools.combinations(X,9):
# 		#	print(s)

# 	#ml of LS
# 	#w = np.linalg.inv(Phi.T.dot(Phi)).dot(Phi.T.dot(y))
# 	#regularized LS
# 	pl1 = plt.plot(X,y,'o',label='Given Data')
# 	for l in np.logspace(-8, -2, num=5):
# 		w_reg = np.linalg.inv(l*np.eye(Phi.shape[1])+Phi.T.dot(Phi)).dot(Phi.T.dot(y))

# 		#w0 = w +0.1*np.random.randn(w.shape[0])
# 		#wgc_batch = gc.numeric_grad_descent(fPhi,w0,lr=0.02, tol=5e-4)
# 		#wgc_sdg, normlist = sdg(fPhi_i,w0)

# 		#print(gc.eval_numerical_gradient_i(fPhi_i,w0,1))

# 		f = np.zeros((xs.shape[0]))
# 		Phip = np.zeros((xs.shape[0],N))
# 		#f_sdg = np.zeros((X.shape[0]))
# 		for ii in range(xs.shape[0]):
# 			Phip[ii,:]=[xs[ii]**i for i in range(N)]
# 			#polynomial
# 			#f[ii] = w.T.dot(Phi[ii])
# 			#cosine
# 			#f[ii] = w.dot([math.cos(X[ii]*np.pi*i) for i in range(1,N+1)])
# 			#regularized
# 			f[ii] = w_reg.dot(Phip[ii])

# 		pl2 = plt.plot(xs,f,label='lambda ='+str(l))
# 		#pl3 = plt.plot(X,f_sdg,'1',label='SGD')
# 		#plt.show()
# 		# #pylab.legend(handles=[pl1,pl2])

# 		plt.xlabel('x')
# 		plt.ylabel('y')
# 	plt.legend(numpoints=1,loc='upper right')
# 	plt.savefig('../../figures/ps3_1s_'+str(N)+'.pdf')
# 	plt.clf()
N=8
Phi = np.zeros((X.shape[0],N))
for ii in range(X.shape[0]):
	#cosine
	Phi[ii,:]=[math.cos(X[ii]*np.pi*i) for i in range(1,N+1)]
	#polynomial
	#Phi[ii,:]=[X[ii]**i for i in range(N)]

w = np.linalg.inv(Phi.T.dot(Phi)).dot(Phi.T.dot(y))
w0 = w +0.1*np.random.randn(w.shape[0])
#wgc_sdg_c, normlist = sdg(fPhicos_i,w0)
print(w)
clf = linear_model.Lasso(alpha = 0.1)
clf.fit(Phi,y)
print(clf.coef_)
clf = linear_model.Ridge(alpha = 0.1)
clf.fit(Phi,y)
print(clf.coef_)

clf = linear_model.LinearRegression()
clf.fit(Phi,y)
print(clf.coef_)
clf = linear_model.LinearRegression()
clf.fit(Phi,y)
#print(clf.residues_)

for N in range (1,9,1):
	Phi = np.zeros((X.shape[0],N))
	for ii in range(X.shape[0]):
		#cosine
		Phi[ii,:]=[math.cos(X[ii]*np.pi*i) for i in range(1,N+1)]
		#polynomial
		#Phi[ii,:]=[X[ii]**i for i in range(N)]
	clf = linear_model.LinearRegression()
	clf.fit(Phi,y)
	#print(clf.residues_)
	(x,r,rk,s) = np.linalg.lstsq(Phi, y)
	print(r)

