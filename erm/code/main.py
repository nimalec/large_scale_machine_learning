import os
import numpy as np
import scipy
import matplotlib
import mnist
import pickle
import time
matplotlib.use('agg')
from matplotlib import pyplot

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

def load_MNIST_dataset():
	PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
	try:
		dataset = pickle.load(open(PICKLE_FILE, 'rb'))
	except:
		# load the MNIST dataset
		mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
		Xs_tr, Lbls_tr = mnist_data.load_training();
		Xs_tr = Xs_tr.transpose() / 255.0
		Ys_tr = np.zeros((10, 60000))
		for i in range(60000):
			Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
		Xs_te, Lbls_te = mnist_data.load_testing();
		Xs_te = Xs_te.transpose() / 255.0
		Ys_te = np.zeros((10, 10000))
		for i in range(10000):
			Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
		dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
		pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
	return dataset

"""Computes the softmax matrix (c * n) corresponding to the 2D matrix X (c * n)"""
def softmax(X):
	c, n = X.shape
	# scale matrix down to avoid overflow
	Z = X - np.amax(X, axis=0)
	mat = np.zeros((c,n))
	sum = np.zeros(n)
	for k in range(c):
		sum = sum + np.exp(Z[k])
	for i in range(c):
		mat[i] = np.true_divide(np.exp(Z[i]), sum)
	return mat

# compute the gradient of the multinomial logistic regression objective, with regularization
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the gradient of the model parameters
def multinomial_logreg_grad(Xs, Ys, gamma, W):
	c, d = W.shape
	grad = np.zeros((c,d))
	product = np.matmul(W,Xs)
	grad = np.matmul((softmax(product)-Ys),Xs.T)
	return (np.true_divide(grad,len(Xs[0])) + gamma*W)

# compute the error of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels

def multinomial_logreg_error(Xs, Ys, W):
	start_logreg_error = time.time()
	c,d = W.shape
	_,n = Xs.shape
	Ys = Ys.astype(int)
	sftmx_preds = softmax(np.matmul(W,Xs))
	preds_arg  = np.argmax(sftmx_preds,axis=0)
	preds = np.zeros([c,n])
	count = 0
	for ii in range(n):
	    preds[preds_arg[ii],ii] = int(1.0)
	    if np.array_equal(preds[:,ii], Ys[:,ii]):
	        count += 1
	error = (n-count)/n;
	assert(error>0)
	end_logreg_error = time.time()
	# print("Logreg Error Time: " + str(end_logreg_error - start_logreg_error))
	return error

# run gradient descent on a multinomial logistic regression objective, with regularization
#
# Xs            training examples (d * n)
# Ys            training labels   (d * c)
# gamma         L2 regularization constant
# W0            the initial value of the parameters (c * d)
# alpha         step size/learning rate
# num_iters     number of iterations to run
# monitor_freq  how frequently to output the parameter vector
#
# returns       a list of models parameters, one every "monitor_freq" iterations
def gradient_descent(Xs, Ys, gamma, W0, alpha, num_iters, monitor_freq):
	grad = lambda Xs, Ys, gamma, W: multinomial_logreg_grad(Xs, Ys, gamma, W)
	W_update = []
	W = W0
	tt = 0
	while tt < num_iters:
		W = W-alpha*grad(Xs, Ys, gamma, W)
		if tt%monitor_freq == 0:
			W_update.append(W)
			tt+=1
		else:
			tt += 1
	return np.array(W_update)

# estimate the error of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
# nsamples  number of samples to use for the estimation
#
# returns   the gradient
def estimate_multinomial_logreg_error(Xs, Ys, W, nsamples):
	start_subsample_error = time.time()
	c,d = W.shape
	_,n = Xs.shape
	sampled_Xs = np.zeros((d, nsamples))
	sampled_Ys = np.zeros((c, nsamples))
	random = np.random.randint(0, high=n, size=nsamples)
	for i in range(nsamples):
		sampled_Xs[:,i] = Xs[:,random[i]]
		sampled_Ys[:,i] = Ys[:,random[i]]
	error = multinomial_logreg_error(sampled_Xs, sampled_Ys, W)
	end_subsample_error = time.time()
	# print("Subsample Error Time: " + str(end_subsample_error - start_subsample_error))
	return error

if __name__ == "__main__":
	(Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
	gamma =  0.0001  # L2 regualarization constant
	alpha =  1.0    # GD step size
	num_iters = 1000
	monitor_freq = 10
	W0 = np.random.rand(len(Ys_tr[:,0]),len(Xs_tr[:,0]))
	# W0 = np.zeros((len(Ys_tr[:,0]),len(Xs_tr[:,0]))) + 1

	W_iter = gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, num_iters, monitor_freq)
