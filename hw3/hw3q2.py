'''
Author: David Antaki
Date: 6/15/22
'''
from datetime import datetime
import random
from time import perf_counter, time
from tracemalloc import start
from torchsummary import summary
import torch.nn.functional as F
import torch.nn as nn
import torch
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from scipy.stats import multivariate_normal as mvn
import scipy
import numpy as np
import matplotlib.colors as mcol
import matplotlib.pyplot as plt  # For general plotting
from sys import float_info  # Threshold smallest positive floating value
from sklearn.model_selection import KFold, cross_val_score
from math import ceil, floor

import matplotlib.pyplot as plt  # For general plotting
import matplotlib.colors as mcol
from matplotlib.ticker import MaxNLocator

import numpy as np
from sklearn.preprocessing import PolynomialFeatures  # Important new include
from sklearn.model_selection import KFold  # Important new include


def eigvalsh_to_eps(spectrum, cond=None, rcond=None):
    """Taken from scipy._multivariate
    """
    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = spectrum.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        cond = factor[t] * np.finfo(t).eps
    eps = cond * np.max(abs(spectrum))
    return eps


def is_pos_def(x):
    s, u = scipy.linalg.eigh(x, lower=True, check_finite=True)
    eps = eigvalsh_to_eps(s, None, None)
    if np.min(s) < -eps:
        return False
    return True


def generate_pos_semidefinite_matrix(n_dims):
    '''
    For generating valid covariance matrices, brute force.
    n_dims: number of dimensions
    '''

    cov_mat = np.ones((n_dims, n_dims))
    while(True):
        for i in range(n_dims):
            for j in range(n_dims):
                # cov_mat[i, j] = random.uniform(-3, 3)
                cov_mat[i, j] = random.uniform(0, 1)
        print(cov_mat)
        print(np.transpose(cov_mat))
        # cov_mat = cov_mat*np.transpose(cov_mat)
        cov_mat = (cov_mat*cov_mat.T)
        if is_pos_def(cov_mat):
            break

        print(cov_mat)
        # input()
    return cov_mat


# print(generate_pos_semidefinite_matrix(10))
# print(sklearn.datasets.make_spd_matrix(10))


def create_data_no_labels(mean, cov, N):
    # Draw random variable samples. Don't assign labels
    # The mvn has to be in a list to generate correct number of dimensions of data from .rvs(), not sure way.
    Gs = [mvn(mean=mean, cov=cov)]
    # N_train samples of n-dimensional samples of random variable x
    X = np.concatenate([G.rvs(size=N) for G in Gs])
    # Will return an X of shape (1, N)
    return np.abs(X)


# Number of dimensions
n = 10
# Number of samples
N_train = 50
N_test = 1000

fig, ax = plt.subplots(figsize=(10, 10))

gmm_pdf = {}
# 'mean_mu' = arbitrary Gaussian with non-zero mean mu and non-diagonal covariance matrix Sigma for a n-dimensional random vector x.
gmm_pdf['mean_mu'] = np.array(
    [-1,  2, - 2,  1, - 2,  1, - 2, - 2,  0, - 3])
gmm_pdf['0_mean'] = np.zeros(n)
# Gaussian distributions covariance matrices
gmm_pdf['cov_sigma'] = np.array([[0.94497952, - 0.42082568, - 0.99666439,  0.18569479, - 0.85582546, 0.11287041,
                                - 0.85177261,  0.91593539,  0.30855983, - 0.37096964],
                                [-0.42082568,  0.97870757, 1.06175119, - 0.19458909, 1.03347265, 0.05726852,
                                0.81388603, - 0.99127419, - 0.26214692,  0.41427894],
                                [-0.99666439, 1.06175119,  2.76378319, - 0.43616616,  2.28496975, - 0.12516731,
                                1.96747988, - 2.17508534, - 0.73858794,  1.0008154],
                                [0.18569479, - 0.19458909, - 0.43616616,  0.65113804, - 0.37752604, - 0.01259725,
                                - 0.2462577,   0.35905941,  0.22037279, - 0.18752859],
                                [-0.85582546,  1.03347265,  2.28496975, - 0.37752604,  2.92334052, - 0.06858065,
                                1.86095023, - 2.37117841, - 0.7003034,   0.9452789],
                                [0.11287041, 0.05726852, - 0.12516731, - 0.01259725, - 0.06858065, 0.59436176,
                                - 0.23459772,  0.13127576,  0.11350214, - 0.09515098],
                                [-0.85177261, 0.81388603,  1.96747988, - 0.2462577, 1.86095023, - 0.23459772,
                                2.41708513, - 1.94458835, - 0.54095705, 0.96050796],
                                [0.91593539, - 0.99127419, - 2.17508534,  0.35905941, - 2.37117841,  0.13127576,
                                - 1.94458835, 2.89031349, 0.57027545, - 0.9039707],
                                [0.30855983, - 0.26214692, - 0.73858794,  0.22037279, - 0.7003034,  0.11350214,
                                - 0.54095705,  0.57027545,  0.87881248, - 0.32184057],
                                [-0.37096964,  0.41427894,  1.0008154, - 0.18752859, 0.9452789, - 0.09515098,
                                0.96050796, - 0.9039707, - 0.32184057, 0.9527596]])


def generate_data(hyper_alpha: float):
    global X_train_x
    global X_train_x_bias
    global Y_train
    global X_test_x
    global X_test_x_bias
    global Y_test
    global gmm_pdf

    # Data
    # Arbitrary non-zero n-dimensional vector 'a'
    # vector_a = np.array([-6,   2,  -9,   9,   6,   4,  -9,   1,  -4,  -2])
    vector_a = np.array([[6], [2],  [9],   [9],   [6],
                        [4],  [9],   [1],  [4],  [2]])
    assert(vector_a.T.shape == (1, n))

    # Mean and covariance of data pdfs conditioned on labels
    # Gaussian distributions means
    gmm_pdf['alpha_I_cov'] = np.identity(n)*hyper_alpha
    gmm_pdf['unit_variance'] = np.identity(n)*hyper_alpha

    ############ Generate training data of (X_train_x, Y_train) pairs ############
    # Draw N_train iid samples of n-dimensional samples of random variable x
    X_train_x = create_data_no_labels(
        gmm_pdf['mean_mu'], gmm_pdf['cov_sigma'], N_train)
    assert(X_train_x.shape == (N_train, n))
    X_train_x_bias = add_bias_term(X_train_x)
    assert(X_train_x_bias.shape == (N_train, n+1))
    # print(X_train_x.shape)
    # Draw N_train iid samples of n-dimensional samples of randomal variable z from 0-mean alpha*I-covariance-matrix
    X_train_z = create_data_no_labels(
        gmm_pdf['0_mean'], gmm_pdf['alpha_I_cov'], N_train)
    assert(X_train_z.shape == (N_train, n))
    # print(X_train_z.shape)
    # Draw N_train iid samples of a scalar random variable v from a 0-mean unit-variance Gaussian pdf.
    X_train_v = np.random.normal(size=(N_train, 1))
    assert(X_train_v.shape == (N_train, 1))
    # Calculate scalar values of a new random variable
    Y_train = np.empty((N_train, 1))
    for i in range(0, N_train):
        Y_train[i] = vector_a.T.dot(X_train_x[i]+X_train_z[i])+X_train_v[i]
    assert(Y_train.shape == (N_train, 1))
    ############ END Generate training data of (X_train_x, Y_train) pairs ############

    ############# Generate testing data of (X_test_x, Y_test) pairs############
    # Generate test dataset in same manner as above
    X_test_x = create_data_no_labels(
        gmm_pdf['mean_mu'], gmm_pdf['cov_sigma'], N_test)
    assert(X_test_x.shape == (N_test, n))
    X_test_x_bias = add_bias_term(X_test_x)
    assert(X_test_x_bias.shape == (N_test, n+1))
    # print(X_test_x.shape)
    # Draw N_test iid samples of n-dimensional samples of randomal variable z from 0-mean alpha*I-covariance-matrix
    X_test_z = create_data_no_labels(
        gmm_pdf['0_mean'], gmm_pdf['alpha_I_cov'], N_test)
    assert(X_test_z.shape == (N_test, n))
    # print(X_test_z.shape)
    # Draw N_test iid samples of a scalar random variable v from a 0-mean unit-variance Gaussian pdf.
    X_test_v = np.random.normal(size=(N_test, 1))
    assert(X_test_v.shape == (N_test, 1))
    # Calculate scalar values of a new random variable
    Y_test = np.empty((N_test, 1))
    for i in range(0, N_test):
        Y_test[i] = vector_a.T.dot(X_test_x[i]+X_test_z[i])+X_test_v[i]
    assert(Y_test.shape == (N_test, 1))
    ############# END Generate testing data of (X_test_x, Y_test) pairs############


def mle_solution_map(X, y, beta):
    return np.linalg.inv(X.T.dot(X)+beta*np.eye(n+1)).dot(X.T).dot(y)


def mse(y_preds, y_true):
    '''
    We are using gaussian distribution for training data, therefore, minimizing MSE works the same as maximizing the log liklihood.
    '''
    # Residual error (X * theta) - y
    error = y_preds - y_true
    # Loss function is MSE
    return np.mean(error ** 2)


def add_bias_term(X):
    '''
    Adds a column with all 1's to the given matrix.
    '''
    # The column to be added
    col = np.ones((1, np.size(X, 0))).T
    # print(col)
    # print(col.shape)
    # Adding column to array using append() method
    X = np.append(X, col, axis=1)
    # Array after adding the column.
    # print(X)
    return X


def negative_2_times_log_likelihood(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    # log_likelihood_elements = np.empty(y_true.shape)
    # print(log_likelihood_elements.shape)
    # for i in (range(0, y_true.shape[0])):
    # log_likelihood_elements[i] = y_true[i] * \
    #     np.log(y_pred[i]) + (1-y_true[i])*np.log(1-y_pred[i])
    log_likelihood_elements = y_true * \
        np.log(y_pred) + (1-y_true)*np.log(1-y_pred)
    return -2*np.sum(log_likelihood_elements)/len(y_true)

# Negative Log Likelihood (NLL) loss


def nll_loss(parameters, X, y, sigma=1):
    mu_pred = X.dot(parameters)

    # Compute log-likelihood function, setting mu=0 as we're estimating it
    log_lld = np.sum(mvn.pdf(y - mu_pred, 0, sigma))

    # Return NLL
    return -log_lld


def beta_hyperparameter_optimization(alpha) -> float:
    '''Here we perform Cross Validation to decide on the best number of perceptrons to use in our MLP.
    Returns the optimal beta value.
    '''

    num_of_betas = 100
    # Range of betas we will try
    # On the order of 10^-9 to 10^4
    n_betas = np.logspace(-9, 8, num=num_of_betas, base=10.0)
    # n_betas = np.linspace(10e-9, 10e4, num=num_of_betas)
    # print(n_betas)
    # n_betas = np.arange(1e-9, 0, 0.0001)
    # np.max(n_betas)

    # Number of folds for CV
    K = 5

    # STEP 1: Partition the dataset into K approximately-equal-sized partitions
    # Shuffles data before doing the division into folds (not necessary, but a good idea)
    kf = KFold(n_splits=K, shuffle=True)

    # Allocate space for CV
    # No need for training loss storage too but useful comparison
    mse_valid_mk = np.empty((num_of_betas, K))
    # Indexed by model m, data partition k
    mse_train_mk = np.empty((num_of_betas, K))

    # STEP 2: Try all polynomial orders between 1 (best line fit) and 21 (big time overfit) M=2
    for beta_i in range(0, num_of_betas):
        # K-fold cross validation
        k = 0
        # NOTE that these subsets are of the TRAINING dataset
        # Imagine we don't have enough data available to afford another entirely separate validation set
        for train_indices, valid_indices in kf.split(X_train_x):
            # Extract the training and validation sets from the K-fold split
            X_train_k = X_train_x_bias[train_indices]
            y_train_k = Y_train[train_indices]
            X_valid_k = X_train_x_bias[valid_indices]
            y_valid_k = Y_train[valid_indices]

            # Get weights
            weights_mk = mle_solution_map(
                X_train_k, y_train_k, n_betas[beta_i])

            # Make predictions on both the training and validation set
            y_train_k_pred = X_train_k.dot(weights_mk)
            y_valid_k_pred = X_valid_k.dot(weights_mk)

            # Record MSE as well for this model and k-fold
            mse_train_mk[beta_i, k] = mse(y_train_k_pred, y_train_k)
            mse_valid_mk[beta_i, k] = mse(y_valid_k_pred, y_valid_k)
            k += 1

    # STEP 3: Compute the average probability of error for that model (based in this case on number of perceptrons
    # Model average CV loss over folds
    mse_train_m = np.mean(mse_train_mk, axis=1)
    mse_valid_m = np.mean(mse_valid_mk, axis=1)

    # print(mse_train_m)
    # print(mse_valid_m)

    # +1 as the index starts from 0 while the beta values start from 1
    optimal_beta = n_betas[np.argmin(mse_valid_m)]
    print("The beta selected to best fit the data without overfitting is beta={} with alpha={}".format(
        optimal_beta, alpha))

    # Graph MSE vs betas
    # plt.xticks(np.logspace(-9, 8, num=20, base=10.0))
    # plt.yticks(np.arange(0, max(y), 2))
    # ax.plot(n_betas, mse_train_m, color="b",
    #         marker="s", label=r"$D_{train}$")
    ax.plot(n_betas, mse_valid_m,
            marker="x", label=r"alpha={}".format(alpha))

    return optimal_beta


def optimize_model_params(opt_beta: float):
    '''
    Trains model with given optimal beta value.
    Returns MSE of how well the model fits the data using the test data.
    '''
    # Train model parameters on entire training dataset
    poly_train_features_cv = PolynomialFeatures(
        degree=1, include_bias=True)
    X_train_poly = poly_train_features_cv.fit_transform(X_train_x)
    weights_mk = mle_solution_map(
        X_train_x_bias, Y_train, opt_beta)

    # Make predictions on test dataset
    y_test_pred = X_test_x_bias.dot(weights_mk)

    # Get the -2 times log likelihood

    return mse(y_test_pred, Y_test)
    # print(negative_2_times_log_likelihood(Y_test, y_test_pred))


def main():
    # Generate range of alpha hyperparameters to test. Alpha is noise term.
    alphas = np.logspace(-3, 3, num=10, base=10.0)
    # print(alphas)
    for a in alphas:
        a = a*np.trace(gmm_pdf['cov_sigma'])/n
    # print(alphas)
    # Space for storing NLL scores i.e. how well a model fit the test data with the given alpha hyperparam.
    nll_alpha = np.empty((len(alphas), 1))
    print(nll_alpha.shape)
    i = 0
    for a in alphas:
        generate_data(a)
        opt_beta = beta_hyperparameter_optimization(a)
        nll_alpha[i] = optimize_model_params(opt_beta)
        i = i+1

    # For the MSE vs betas
    ax.legend(loc='lower right', shadow=True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlabel("Beta Hyperparameter")
    plt.ylabel("MSE")
    plt.title("MSE vs Beta Hyperparameter")
    plt.grid()
    # plt.show()
    plt.savefig("{}-MSE vs Beta Hyperparameter.png".format(
        datetime.now().strftime("%Y-%d-%m-%H-%M")), dpi=300)
    # Plot NLL scores vs Alpha.
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.set_xscale('log')
    ax2.plot(nll_alpha, alphas, color="b",
            marker="s", label=r"$D_{test}$")
    ax2.legend(loc='upper left', shadow=True)
    plt.xlabel("Alpha Hyperparameter")
    plt.ylabel("MSE")
    plt.title("MSE of Test Dataset vs Alpha")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
