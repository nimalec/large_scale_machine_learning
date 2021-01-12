#!/usr/bin/env python3
import os
import numpy as np
from numpy import random
import scipy
import matplotlib
import mnist
import pickle
import time
import math
matplotlib.use('agg')
from matplotlib import pyplot

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables


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
        # shuffle the training data
        np.random.seed(8675309)
        perm = np.random.permutation(60000)
        Xs_tr = np.ascontiguousarray(Xs_tr[:,perm])
        Ys_tr = np.ascontiguousarray(Ys_tr[:,perm])
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = np.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = np.ascontiguousarray(Xs_te)
        Ys_te = np.ascontiguousarray(Ys_te)
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

# compute the gradient of the multinomial logistic regression objective, with regularization (SAME AS PROGRAMMING ASSIGNMENT 2)
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# ii        the list/vector of indexes of the training example to compute the gradient with respect to
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the average gradient of the regularized loss of the examples in vector ii with respect to the model parameters
def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W):
    c, d = W.shape
    new_Xs = np.empty((d,len(ii)))
    new_Ys = np.empty((c,len(ii)))
    # create subset of the examples and labels
    # with indexes in vector ii
    for idx, j in enumerate(ii):
        new_Xs[:,idx] = Xs[:,int(j)]
        new_Ys[:,idx] = Ys[:,int(j)]
    grad = np.zeros((c,d))
    product = np.matmul(W,new_Xs)
    grad = np.matmul((softmax(product)-new_Ys),new_Xs.T)
    return (np.true_divide(grad,len(ii)) + gamma*W)

# compute the error of the classifier (SAME AS PROGRAMMING ASSIGNMENT 1)
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
    assert(error>=0.)
    end_logreg_error = time.time()
    # print("Logreg Error Time: " + str(end_logreg_error - start_logreg_error))
    return error

# compute the cross-entropy loss of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss(Xs, Ys, gamma, W):
    sftmx_preds = softmax(np.matmul(W,Xs))
    loss = np.sum(-Ys*np.log(sftmx_preds),axis=(0,1)) + np.sum(gamma*W,axis=(0,1))
    return loss/Ys.shape[1]

# gradient descent (SAME AS PROGRAMMING ASSIGNMENT 1)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# num_epochs      number of epochs (passes through the training set, or equivalently iterations of gradient descent) to run
# monitor_period  how frequently, in terms of epochs/iterations to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" epochs
def gradient_descent(Xs, Ys, gamma, W0, alpha, num_epochs, monitor_period):
    start = time.time()
    grad = lambda Xs, Ys, gamma, W: multinomial_logreg_grad_i(Xs, Ys, range(n), gamma, W)
    _,n = Xs.shape
    W_update = []
    W = W0
    tt = 0
    while tt < num_epochs:
        W = W-alpha*grad(Xs, Ys, gamma, W)
        if tt%monitor_period == 0:
            W_update.append(W)
            tt+=1
        else:
            tt += 1
    print("GD Time: " + str(time.time() - start))
    return np.array(W_update)

# gradient descent with nesterov momentum
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# num_epochs      number of epochs (passes through the training set, or equivalently iterations of gradient descent) to run
# monitor_period  how frequently, in terms of epochs/iterations to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" epochs
def gd_nesterov(Xs, Ys, gamma, W0, alpha, beta, num_epochs, monitor_period):
    start = time.time()
    _,n = Xs.shape
    grad = lambda Xs, Ys, gamma, W: multinomial_logreg_grad_i(Xs, Ys, range(n), gamma, W)
    W_update = []
    W = W0
    V_old = W0
    iteration = 0
    for t in range(num_epochs):
        V_new = W - alpha*grad(Xs, Ys, gamma, W)
        W = V_new + beta*(V_new-V_old)
        V_old = V_new
        if t%monitor_period == 0:
            W_update.append(W)
        t += 1
    print("Nesterov Time: " + str(time.time() - start))
    return np.array(W_update)

# SGD: run stochastic gradient descent with minibatching and sequential sampling order (SAME AS PROGRAMMING ASSIGNMENT 2)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def sgd_minibatch_sequential_scan(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    start = time.time()
    _,n = Xs.shape
    grad = lambda Xs, Ys, ii, gamma, W: multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
    W_update = []
    W = W0
    iteration = 0
    for t in range(num_epochs):
        for i in range(int(n/B)):
            ii = []
            start_index = np.random.randint(0, high=n)
            for b in range(B):
                ind = (start_index + b)%n
                ii.append(ind)
            if iteration%monitor_period == 0:
                W_update.append(W)
                iteration+=1
            W = W-alpha*grad(Xs, Ys, np.array(ii), gamma, W)
    print("SGD Time: " + str(time.time() - start))
    return np.array(W_update)

# SGD + Momentum: add momentum to the previous algorithm
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def sgd_mss_with_momentum(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, monitor_period):
    start = time.time()
    _,n = Xs.shape
    c,d = W0.shape
    grad = lambda Xs, Ys, ii, gamma, W: multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
    W_update = []
    W = W0
    V = np.zeros((c,d))
    iter = 0
    for t in range(num_epochs):
        for i in range(int(n/B)):
            ii = []
            start_index = np.random.randint(0, high=n)
            for b in range(B):
                ind = (start_index + b)%n
                ii.append(ind)
            if iter%monitor_period == 0:
                W_update.append(W)
                iter+=1
            V = beta*V - alpha*grad(Xs, Ys, np.array(ii), gamma, W)
            W = W + V
    print("SGD Nesterov Time: " + str(time.time() - start))
    return np.array(W_update)

# Adam Optimizer
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# rho1            first moment decay rate ρ1
# rho2            second moment decay rate ρ2
# B               minibatch size
# eps             small factor used to prevent division by zero in update step
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def adam(Xs, Ys, gamma, W0, alpha, rho1, rho2, B, eps, num_epochs, monitor_period):
    start = time.time()
    _,n = Xs.shape
    c,d = W0.shape
    grad = lambda Xs, Ys, ii, gamma, W: multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
    W_update = []
    W = W0
    iter = 0
    S = np.zeros((c,d))
    R = np.zeros((c,d))
    for t in range(num_epochs):
        for i in range(int(n/B)):
            ii = []
            start_index = np.random.randint(0, high=n)
            for b in range(B):
                ind = (start_index + b)%n
                ii.append(ind)
            if iter%monitor_period == 0:
                W_update.append(W)
            iter += 1
            G = grad(Xs, Ys, np.array(ii), gamma, W)
            S = rho1*S + (1-rho1)*G
            R = rho2*R + (1-rho2)*G**2
            Shat  =  S/(1-rho1**(iter))
            Rhat  =  R/(1-rho2**(iter))
            W  = W -(alpha/(np.sqrt(Rhat + eps)))*Shat
    print("ADAM Time: " + str(time.time() - start))
    return np.array(W_update)

if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()

    # shrink data set for faster debugging
    Xs_tr = Xs_tr[:,:600]
    Ys_tr = Ys_tr[:,:600]
    Xs_te = Xs_te[:,:600]
    Ys_te = Ys_te[:,:600]

    # _________ PART 1 ___________

    gamma = 0.0001
    alpha = 1.0
    num_epochs = 100
    monitor_period = 1
    beta1 = 0.9
    beta2 = 0.99

    W0 = np.random.rand(len(Ys_tr[:,0]),len(Xs_tr[:,0]))

    GD_W_iter = gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, num_epochs, monitor_period)
    Nesterov1_W_iter = gd_nesterov(Xs_tr, Ys_tr, gamma, W0, alpha, beta1, num_epochs, monitor_period)
    Nesterov2_W_iter = gd_nesterov(Xs_tr, Ys_tr, gamma, W0, alpha, beta2, num_epochs, monitor_period)

    GD_train_err = []
    Nesterov1_train_err = []
    Nesterov2_train_err = []
    GD_experiment1 = []
    Nesterov_experiment1 = []

    GD_test_err = []
    Nesterov1_test_err = []
    Nesterov2_test_err = []
    GD_loss = []
    Nesterov1_loss = []
    Nesterov2_loss  = []

    for i in range(GD_W_iter.shape[0]):
        GD_train_err.append(multinomial_logreg_error(Xs_tr, Ys_tr, GD_W_iter[i]))
        Nesterov1_train_err.append(multinomial_logreg_error(Xs_tr, Ys_tr, Nesterov1_W_iter[i]))
        Nesterov2_train_err.append(multinomial_logreg_error(Xs_tr, Ys_tr, Nesterov2_W_iter[i]))
        GD_test_err.append(multinomial_logreg_error(Xs_te, Ys_te, GD_W_iter[i]))
        Nesterov1_test_err.append(multinomial_logreg_error(Xs_te, Ys_te, Nesterov1_W_iter[i]))
        Nesterov2_test_err.append(multinomial_logreg_error(Xs_te, Ys_te, Nesterov2_W_iter[i]))
        GD_loss.append(multinomial_logreg_loss(Xs_tr, Ys_tr,gamma,GD_W_iter[i]))
        Nesterov1_loss.append(multinomial_logreg_loss(Xs_tr, Ys_tr,gamma ,Nesterov1_W_iter[i]))
        Nesterov2_loss.append(multinomial_logreg_loss(Xs_tr, Ys_tr,gamma ,Nesterov2_W_iter[i]))

    # _________ PART 2 ___________

    alpha2 = 0.2
    B = 60
    num_epochs = 10
    monitor_period = 1

    SGD_W_iter = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha2, B, num_epochs, monitor_period)
    SGD_Nesterov1_W_iter = sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha2, beta1, B, num_epochs, monitor_period)
    SGD_Nesterov2_W_iter = sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha2, beta2, B, num_epochs, monitor_period)

    SGD_train_err = []
    SGD_Nesterov1_train_err = []
    SGD_Nesterov2_train_err = []
    SGD_test_err = []
    SGD_Nesterov1_test_err = []
    SGD_Nesterov2_test_err = []
    SGD_loss = []
    SGD_Nesterov1_loss = []
    SGD_Nesterov2_loss = []

    print("SGD_W_iter size: " + str(SGD_W_iter.shape[0]))
    print("SGD_Nesterov1_W_iter size: " + str(SGD_Nesterov1_W_iter.shape[0]))
    print("SGD_Nesterov2_W_iter size: " + str(SGD_Nesterov2_W_iter.shape[0]))

    for i in range(SGD_W_iter.shape[0]):
        SGD_train_err.append(multinomial_logreg_error(Xs_tr, Ys_tr, SGD_W_iter[i]))
        SGD_Nesterov1_train_err.append(multinomial_logreg_error(Xs_tr, Ys_tr, SGD_Nesterov1_W_iter[i]))
        SGD_Nesterov2_train_err.append(multinomial_logreg_error(Xs_tr, Ys_tr, SGD_Nesterov2_W_iter[i]))
        SGD_test_err.append(multinomial_logreg_error(Xs_te, Ys_te, SGD_W_iter[i]))
        SGD_Nesterov1_test_err.append(multinomial_logreg_error(Xs_te, Ys_te, SGD_Nesterov1_W_iter[i]))
        SGD_Nesterov2_test_err.append(multinomial_logreg_error(Xs_te, Ys_te, SGD_Nesterov2_W_iter[i]))
        SGD_loss.append(multinomial_logreg_loss(Xs_tr, Ys_tr,gamma ,SGD_W_iter[i]))
        SGD_Nesterov1_loss.append(multinomial_logreg_loss(Xs_tr, Ys_tr,gamma ,SGD_Nesterov1_W_iter[i]))
        SGD_Nesterov2_loss.append(multinomial_logreg_loss(Xs_tr, Ys_tr,gamma ,SGD_Nesterov2_W_iter[i]))

    # _________ PART 3 ___________

    alpha3 = 0.01
    rho1 = 0.9
    rho2 = 0.999
    epsilon = 1.e-5

    ADAM_W_iter = adam(Xs_tr, Ys_tr, gamma, W0, alpha3, rho1, rho2, B, epsilon, num_epochs, monitor_period)

    ADAM_train_err = []
    ADAM_test_err = []
    ADAM_loss = []

    print("ADAM_W_iter size: " + str(ADAM_W_iter.shape[0]))

    for i in range(ADAM_W_iter.shape[0]):
        ADAM_train_err.append(multinomial_logreg_error(Xs_tr, Ys_tr, ADAM_W_iter[i]))
        ADAM_test_err.append(multinomial_logreg_error(Xs_te, Ys_te, ADAM_W_iter[i]))
        ADAM_loss.append(multinomial_logreg_loss(Xs_tr, Ys_tr,gamma ,ADAM_W_iter[i]))
