#!/usr/bin/env python3
import os
import numpy
from numpy import random
import scipy
from scipy import special
import matplotlib
import mnist
import pickle
import math
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.animation as animation

from tqdm import tqdm
from scipy.special import softmax

import tensorflow as tf
sess = tf.Session()

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

### hyperparameter settings and other constants
### end hyperparameter settings

def load_MNIST_dataset_with_validation_split():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = numpy.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        # shuffle the training data
        numpy.random.seed(8675309)
        perm = numpy.random.permutation(60000)
        Xs_tr = numpy.ascontiguousarray(Xs_tr[:,perm])
        Ys_tr = numpy.ascontiguousarray(Ys_tr[:,perm])
        # extract out a validation set
        Xs_va = Xs_tr[:,50000:60000]
        Ys_va = Ys_tr[:,50000:60000]
        Xs_tr = Xs_tr[:,1:50000]
        Ys_tr = Ys_tr[:,1:50000]
        # load test data
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = numpy.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = numpy.ascontiguousarray(Xs_te)
        Ys_te = numpy.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_va, Ys_va, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset

# compute the cumulative distribution function of a standard Gaussian random variable
def gaussian_cdf(u):
    return 0.5*(1.0 + tf.math.erf(u/numpy.sqrt(2.0)))

# compute the probability mass function of a standard Gaussian random variable
def gaussian_pmf(u):
    return tf.math.exp(-u**2/2.0)/numpy.math.sqrt(2.0*numpy.pi)


# compute the Gaussian RBF kernel matrix for a vector of data points (in TensorFlow)
#
# Xs        points at which to compute the kernel (size: d x m)
# Zs        other points at which to compute the kernel (size: d x n)
# gamma     gamma parameter for the RBF kernel
#
# returns   an (m x n) matrix Sigma where Sigma[i,j] = K(Xs[:,i], Zs[:,j])
def rbf_kernel_matrix(Xs, Zs, gamma):
     m  = np.size(Xs[1,:])
     n =  np.size(Zs[1,:])
     K_temp = np.zeros(m,n)
     for ii in range(m):
         for jj in range(n):
              K_temp[ii,jj]= np.norm(X[:,ii]-Z[:,jj])
     K = tf.exp(-gamma*tf.Tensor(K_temp))
     return K


# compute the distribution predicted by a Gaussian process that uses an RBF kernel (in TensorFlow)
#
# Xs            points at which to compute the kernel (size: d x n) where d is the number of parameters
# Ys            observed value at those points (size: n)
# gamma         gamma parameter for the RBF kernel
# sigma2_noise  the variance sigma^2 of the additive gaussian noise used in the model
#
# returns   a function that takes a value Xtest (size: d) and returns a tuple (mean, variance)
# def gp_prediction(Xs, Ys, gamma, sigma2_noise):
#     sig=lambda x, z: rbf_kernel_matrix(x,z,gamma)
#     def prediction_mean_and_variance(Xtest):
#         mean = tf.matmul(tf.transpose(sig(Xs,Xtest)), tf.linalg.matmul(tf.linalg.inv(sig(Xs,Xs)),Ys))
#         variance = sig(Xtest,Xtest) - tf.linalg.matmul(tf.transpose(sig(Xtest,Xs)), tf.linalg.matmul(tf.linalg.inverse(sig(Xs,Xs)), sig(Xtest,Xs))
#         pred = (mean, variance)
#         return pred
#         prediction_mean_and_variance= lambda Xtest: prediction_mean_and_variance(Xtest)
#     return prediction_mean_and_variance
#
# # compute the probability of improvement (PI) acquisition function
# #
# # Ybest     points at which to compute the kernel (size: d x n)
# # mean      mean of prediction
# # stdev     standard deviation of prediction (the square root of the variance)
# #
# # returns   PI acquisition function
def pi_acquisition(Ybest, mean, stdev):
    pi_acq = lambda Ybest, mean, stdev: -scipy.stats.norm.cdf((Ybest-mean)/stdev)
    return pi_acq
# #compute the expected improvement (EI) acquisition function
# #
# # Ybest     points at which to compute the kernel (size: d x n)
# # mean      mean of prediction
# # stdev     standard deviation of prediction
# #
# # returns   EI acquisition function
def ei_acquisition(Ybest, mean, stdev):
    ei_acq =  lambda Ybest, mean, stdev: -(scipy.stats.norm.pdf((Ybest- mean)/stdev) + ((Ybest- mean)/stdev)*scipy.stats.norm.cdf((Ybest- mean)/stdev))
    return ei_acq
# # return a function that computes the lower confidence bound (LCB) acquisition function
# #
# # kappa     parameter for LCB
# #
# # returns   function that computes the LCB acquisition function
def lcb_acquisition(kappa):
    def A_lcb(Ybest, mean, stdev):
        A_lcb = lambda Ybest, mean, stdev: mean - kappa*stdev
        return A_lcb
    return A_lcb

