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
    x_sq = tf.norm(Xs,ord=2,axis=0)**2
    z_sq = tf.norm(Zs,ord=2,axis=0)**2
    x_sq = x_sq[:,tf.newaxis]
    # z_sq = z_sq[tf.newaxis,:]
    z_sq = z_sq
    # print("Xs.T shape " + str(tf.transpose(Xs).get_shape()) + "\n\n")
    # print("Zs shape " + str(Zs.shape) + "\n\n")
    temp = x_sq + z_sq - 2*tf.linalg.matmul(tf.transpose(Xs),Zs)
    sigma = tf.exp(-gamma*temp)
    # return tf.dtypes.cast(sigma,dtype=tf.dtypes.float64)
    return sigma

    # for i in range(m):
    #     for j in range(n):
    #         sigma = -gamma*numpy.linalg.norm(Xs[:,i] - Zs[:,j])
    #         print("Sigma = ")
    #         print(str(sigma) + "\n")
    #         Sigma[i,j] = -gamma*tf.norm(Xs[:,i] - Zs[:,j])

# compute the distribution predicted by a Gaussian process that uses an RBF kernel (in TensorFlow)
#
# Xs            points at which to compute the kernel (size: d x n) where d is the number of parameters
# Ys            observed value at those points (size: n)
# gamma         gamma parameter for the RBF kernel
# sigma2_noise  the variance sigma^2 of the additive gaussian noise used in the model
#
# returns   a function that takes a value Xtest (size: d) and returns a tuple (mean, variance)
def gp_prediction(Xs, Ys, gamma, sigma2_noise):
    sig=lambda x, z: rbf_kernel_matrix(x,z,gamma)
    pre_inv = sig(Xs,Xs)+sigma2_noise*tf.eye(Xs.shape[1],dtype=tf.dtypes.float64)
    inv = tf.linalg.inv(pre_inv)

    prod = tf.matmul(inv,Ys)
    def prediction_mean_and_variance(Xtest):
        k_star = sig(Xs,Xtest)

        mean = tf.matmul(tf.transpose(k_star), prod)
        # op1 = sig(Xtest,Xtest)
        # op2 = tf.transpose(k_star)
        #
        # print("inverse shape " + str(inv.get_shape()) + "\n\n")
        # print("k_star shape " + str(k_star.get_shape()) + "\n\n")
        # print("sig(Xtest,Xtest) shape " + str(op1.get_shape()) + "\n\n")
        # print("tf.transpose(k_star) shape " + str(op2.get_shape()) + "\n\n")
        # print("inv*k_star shape ")
        # op3 = tf.linalg.matvec(inv, tf.reshape(k_star,shape=[-1]))
        # print(str(op3.get_shape()) + "\n\n")
        # op4 = tf.linalg.matvec(tf.transpose(k_star), tf.linalg.matvec(inv, tf.reshape(k_star,shape=[-1])))
        # print("k_star.T*inv*k_star shape " + str(op4.get_shape()) + "\n\n")


        variance = sig(Xtest[:,tf.newaxis],Xtest[:,tf.newaxis]) + sigma2_noise - tf.linalg.matvec(tf.transpose(k_star), tf.linalg.matvec(inv, tf.reshape(k_star,shape=[-1])))
        return (mean, variance)
    return prediction_mean_and_variance

# compute the probability of improvement (PI) acquisition function
#
# Ybest     points at which to compute the kernel (size: d x n)
# mean      mean of prediction
# stdev     standard deviation of prediction (the square root of the variance)
#
# returns   PI acquisition function
def pi_acquisition(Ybest, mean, stdev):
    return -gaussian_cdf((Ybest-mean)/stdev)

# compute the expected improvement (EI) acquisition function
#
# Ybest     points at which to compute the kernel (size: d x n)
# mean      mean of prediction
# stdev     standard deviation of prediction
#
# returns   EI acquisition function
def ei_acquisition(Ybest, mean, stdev):
    return -(gaussian_pmf((Ybest- mean)/stdev)
            + ((Ybest- mean)/stdev)*gaussian_cdf((Ybest- mean)/stdev))*stdev

# return a function that computes the lower confidence bound (LCB) acquisition function
#
# kappa     parameter for LCB
#
# returns   function that computes the LCB acquisition function
def lcb_acquisition(Ybest, mean, stdev):
    return mean - 2*stdev

# gradient descent to do the inner optimization step of Bayesian optimization
#
# objective     the objective function to minimize, as a function that takes a tensorflow variable and returns an expression
# d             the dimension of the input
# alpha         learning rate/step size
# num_iters     number of iterations of gradient descent
#
# returns       a function that takes input
#   x0            initial value to assign to variable x
#               and runs gradient descent, and
#   returns     (obj_min, x_min), where
#       obj_min     the value of the objective after running iterations of gradient descent
#       x_min       the value of x after running iterations of gradient descent
def gradient_descent(objective, d, alpha, num_iters):
    # construct the tensorflow graph associated with this objective
    x = tf.Variable(numpy.zeros((d,1)))
    # print("x shape " + str(x.get_shape()) + "\n\n")
    f = objective(x)
    (g, ) = tf.gradients(f, [x])
    sess.run(tf.global_variables_initializer())
    gd_step = x.assign(x - alpha * g)
    # a function that computes gradient descent
    def gd_from_initial_value(x0):
        with sess.as_default():
            x.assign(x0).eval()
            for it in range(num_iters):
                gd_step.eval()
            return (f.eval().item(), x.eval())
    return gd_from_initial_value

def acq_helper(y_best, prediction_mean_and_variance, acquisition):
    def acq (x):
        (mean,variance) = prediction_mean_and_variance(x)
        std_dev = tf.math.sqrt(variance)
        temp = acquisition(y_best, mean, std_dev)
        return temp
    return acq

# run Bayesian optimization to minimize an objective
#
# objective     objective function
# d             dimension to optimize over
# gamma         gamma to use for RBF hyper-hyperparameter
# sigma2_noise  additive Gaussian noise parameter for Gaussian Process
# acquisition   acquisition function to use (e.g. ei_acquisition)
# random_x      function that returns a random sample of the parameter we're optimizing over (e.g. for use in warmup)
# gd_nruns      number of random initializations we should use for gradient descent for the inner optimization step
# gd_alpha      learning rate for gradient descent
# gd_niters     number of iterations for gradient descent
# n_warmup      number of initial warmup evaluations of the objective to use
# num_iters     number of outer iterations of Bayes optimization to run (including warmup)
#
# returns       tuple of (y_best, x_best, Ys, Xs), where
#   y_best          objective value of best point found
#   x_best          best point found
#   Ys              vector of objective values for all points searched (size: num_iters)
#   Xs              matrix of all points searched (size: d x num_iters)
def bayes_opt(objective, d, gamma, sigma2_noise, acquisition, random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters):
    #initialize optimization objective
    y_best = float("inf")
    # warmup phase to get priors
    Xs = numpy.zeros((d,num_iters))
    Ys = numpy.zeros((num_iters,1))
    #determine new y_best for each warmup step => obtain gaussian prior
    for i in range(n_warmup):
        x_i = random_x()
        Xs[:,i] = x_i
        y_i = objective(x_i)
        Ys[i,0] = y_i
        if y_i <= y_best:
           x_best = x_i
           y_best = y_i
    # optimization
    x_star =tf.zeros([d,1])
    for i in range(n_warmup+1,num_iters):
        prediction_mean_and_variance = gp_prediction(Xs, Ys, gamma, sigma2_noise) #function to make predicitons
        acq = acq_helper(y_best, prediction_mean_and_variance, acquisition)
        gd = gradient_descent(acq,d,gd_alpha,gd_niters)
        # Run over gradient steps
        for j in range(gd_nruns):
            x_star = random_x()
            (y_min, x_min) = gd(x_star)
            Xs[:,i] = x_min
            Ys[i,0] = y_min
            if y_min <= y_best:
                x_best = x_min
                y_best = y_min
    return (y_best, x_best, Ys, Xs)

# a one-dimensional test objective function on which to run Bayesian optimization
def test_objective(x):
    return (numpy.cos(8.0*x) - 0.3 + (x-0.5)**2)

# produce an animation of the predictions made by the Gaussian process in the course of 1-d Bayesian optimization
#
# objective     objective function
# gamma         gamma to use for RBF hyper-hyperparameter
# sigma2_noise  additive Gaussian noise parameter for Gaussian Process
# Ys            vector of objective values for all points searched (size: num_iters)
# Xs            matrix of all points searched (size: d x num_iters)
# xs_eval       list of xs at which to evaluate the mean and variance of the prediction at each step of the algorithm
# filename      path at which to store .mp4 output file
def animate_predictions(objective, gamma, sigma2_noise, Ys, Xs, xs_eval, filename):
    mean_eval = []
    variance_eval = []
    for it in range(len(Ys)):
        print("rendering frame %i" % it)
        Xsi = Xs[:, 0:(it+1)]
        Ysi = Ys[0:(it+1),:]
        gp_pred = gp_prediction(Xsi, Ysi, gamma, sigma2_noise)
        pred_means = []
        pred_variances = []
        np_XE = numpy.expand_dims(numpy.zeros(Xs[:,0].shape),axis=1)
        # print("XE shape = " + str(np_XE.shape))
        XE = tf.Variable(numpy.expand_dims(numpy.zeros(Xs[:,0].shape),axis=1))
        (pred_mean, pred_variance) = gp_pred(XE)
        with sess.as_default():
            for x_eval in xs_eval:
                # print("XE reassign shape = " + str(numpy.expand_dims(numpy.array([x_eval]),axis=1).shape))

                XE.assign(numpy.expand_dims(numpy.array([x_eval]),axis=1)).eval()
                pred_means.append(pred_mean.eval().item())
                pred_variances.append(pred_variance.eval().item())
        mean_eval.append(numpy.array(pred_means))
        variance_eval.append(numpy.array(pred_variances))

    fig, ax = pyplot.subplots()

    def anim_init():
        fig.clear()

    def animate(i):
        ax = fig.gca()
        ax.clear()
        ax.fill_between(xs_eval, mean_eval[i] + 2.0*numpy.sqrt(variance_eval[i]), mean_eval[i] - 2.0*numpy.sqrt(variance_eval[i]), color="#eaf1f7")
        ax.plot(xs_eval, objective(xs_eval))
        ax.plot(xs_eval, mean_eval[i], color="r")
        ax.scatter(Xs[0,0:(i+1)], Ys[0:(i+1)])
        pyplot.title("Bayes Opt After %d Steps" % (i+1))
        pyplot.xlabel("parameter")
        pyplot.ylabel("objective")
        fig.show()

    ani = animation.FuncAnimation(fig, animate, frames=range(len(Ys)), init_func=anim_init, interval=400, repeat_delay=1000)

    ani.save(filename)

# compute the gradient of the multinomial logistic regression objective, with regularization (SAME AS PROGRAMMING ASSIGNMENT 3)
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
    new_Xs = numpy.empty((d,len(ii)))
    new_Ys = numpy.empty((c,len(ii)))
    # create subset of the examples and labels
    # with indexes in vector ii
    for idx, j in enumerate(ii):
        new_Xs[:,idx] = Xs[:,int(j)]
        new_Ys[:,idx] = Ys[:,int(j)]
    grad = numpy.zeros((c,d))
    product = numpy.matmul(W,new_Xs)
    grad = numpy.matmul((softmax(product)-new_Ys),new_Xs.T)
    return (numpy.true_divide(grad,len(ii)) + gamma*W)

# compute the error of the classifier (SAME AS PROGRAMMING ASSIGNMENT 3)
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
    sftmx_preds = softmax(numpy.matmul(W,Xs))
    preds_arg  = numpy.argmax(sftmx_preds,axis=0)
    preds = numpy.zeros([c,n])
    count = 0
    for ii in range(n):
        preds[preds_arg[ii],ii] = int(1.0)
        if numpy.array_equal(preds[:,ii], Ys[:,ii]):
            count += 1
    error = (n-count)/n;
    assert(error>=0.)
    end_logreg_error = time.time()
    # print("Logreg Error Time: " + str(end_logreg_error - start_logreg_error))
    return error

# compute the cross-entropy loss of the classifier (SAME AS PROGRAMMING ASSIGNMENT 3)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss(Xs, Ys, gamma, W):
    sftmx_preds = softmax(numpy.matmul(W,Xs))
    loss = numpy.sum(-Ys*numpy.log(sftmx_preds),axis=(0,1)) + numpy.sum(gamma*W,axis=(0,1))
    return loss/Ys.shape[1]

# SGD + Momentum: run stochastic gradient descent with minibatching, sequential sampling order, and momentum (SAME AS PROGRAMMING ASSIGNMENT 3)
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
    V = numpy.zeros((c,d))
    iter = 0
    for t in range(num_epochs):
        for i in range(int(n/B)):
            ii = []
            start_index = numpy.random.randint(0, high=n)
            for b in range(B):
                ind = (start_index + b)%n
                ii.append(ind)
            if iter%monitor_period == 0:
                W_update.append(W)
            iter+=1
            V = beta*V - alpha*grad(Xs, Ys, numpy.array(ii), gamma, W)
            W = W + V
    print("SGD Nesterov Time: " + str(time.time() - start))
    return W_update

# produce a function that runs SGD+Momentum on the MNIST dataset, initializing the weights to zero
#
# mnist_dataset         the MNIST dataset, as returned by load_MNIST_dataset_with_validation_split
# num_epochs            number of epochs to run for
# B                     the batch size
#
# returns               a function that takes parameters
#   params                  a numpy vector of shape (3,) with entries that determine the hyperparameters, where
#       gamma = 10^(-8 * params[0])
#       alpha = 0.5*params[1]
#       beta = params[2]
#                       and returns (the validation error of the final trained model after all the epochs) minus 0.9.
#                       if training diverged (i.e. any of the weights are non-finite) then return 0.1, which corresponds to an error of 1.
def mnist_sgd_mss_with_momentum(mnist_dataset, num_epochs, B):
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    W0 = numpy.full((len(Ys_tr[:,0]),len(Xs_tr[:,0])),0.)


if __name__ == "__main__":
    # Part 2: Synthetic Objective

    gamma = 10
    sigma2_noise = 0.001
    d = 1
    random.seed(1)
    def random_x ():
        return numpy.random.rand(d,1)
    gd_nruns = 5
    gd_niters = 100
    gd_alpha = 0.01
    n_warmup = 3
    num_iters = 20

    print("# # # Part 2.1: running synthetic objective on all 3 acquisition functions # # # ")

    (y_best, x_best, Ys, Xs) = bayes_opt(test_objective, d, gamma, sigma2_noise, pi_acquisition, random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters)
    print("\nProbability of Improvement: \
            Best Parameter = " + str(x_best) + " : Objective Value = " + str(y_best))

    # ei_acquisition(Ybest, mean, stdev):
    (y_best, x_best, Ys, Xs) = bayes_opt(test_objective, d, gamma, sigma2_noise, ei_acquisition, random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters)
    print("\nExpected Improvement: \
            Best Parameter = " + str(x_best) + " : Objective Value = " + str(y_best))

    # lcb_acquisition(kappa):
    (y_best, x_best, Ys, Xs) = bayes_opt(test_objective, d, gamma, sigma2_noise, lcb_acquisition, random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters)
    print("\nLower Confidence Bound: \
            Best Parameter = " + str(x_best) + " : Objective Value = " + str(y_best))

    print("\n\n# # # Part 2.2: animation # # # ")

    xs_eval = numpy.arange(-.5,1.55,0.05)
    filename = "video.mp4"
    # print("Xs shape = " + str(Xs.shape))
    # print("Ys shape = " + str(Ys.shape))

    # animate_predictions(test_objective, gamma, sigma2_noise, Ys, Xs, xs_eval, filename)

    print("# # # Part 2.3: changing gamma # # # ")
