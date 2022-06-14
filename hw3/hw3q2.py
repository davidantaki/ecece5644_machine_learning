import random
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
from sklearn.model_selection import KFold


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
print(sklearn.datasets.make_spd_matrix(10))


def create_data_no_labels(mean, cov, N):
    # Draw random variable samples. Don't assign labels
    # The mvn has to be in a list to generate correct number of dimensions of data from .rvs(), not sure way.
    Gs = [mvn(mean=mean, cov=cov)]
    # N_train samples of n-dimensional samples of random variable x
    X = np.concatenate([G.rvs(size=N_train) for G in Gs])
    # Will return an X of shape (1, N)
    return X

############################# GENERATE DATA #############################


# Number of dimensions
n = 10
# Number of samples
N_train = 50
N_test = 1000
# Parameters
alpha = 0.4

# Data
# Arbitrary non-zero n-dimensional vector 'a'
vector_a = np.array([[-6,   2,  -9,   9,   6,   4,  -9,   1,  -4,  -2, ],
                     [-2,   4,  -1,   2, -10,  -5,   8,   0,   9,   0, ],
                     [-1,   4, -10,  -4,   7,  -2,  -1,  -6,   8,  -4, ],
                     [-8,   0,   3,   8,  -8,   9,   1,   2,   2,   2, ],
                     [-1,  -7,   7,   2,   9,  -5,  -1,   3,   1,  -5, ],
                     [-5,   1,   8,   6,   4,   8,  -4,   6,  -6,   5, ],
                     [-6,   4,  -8,  -1,   5,   9,  -7,  -1,   0,   3, ],
                     [-9,  -5,   8,   0,   3,   9,   3,  -1,  -7,  -1, ],
                     [9,   2,   6,  -4,   3,   2,   7,  -9,  -9,  -9, ],
                     [5,   1,   0,   1,  -2,   1,   5, -10,   8,   9, ]])

gmm_pdf = {}
# Class priors
gmm_pdf['priors'] = np.array([0.5, 0.5, 0.5, 0.5])
# Mean and covariance of data pdfs conditioned on labels
# Gaussian distributions means
# 'mean_mu' = arbitrary Gaussian with non-zero mean mu and non-diagonal covariance matrix Sigma for a n-dimensional random vector x.
gmm_pdf['mean_mu'] = np.array([-1,  2, - 2,  1, - 2,  1, - 2, - 2,  0, - 3])
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
gmm_pdf['alpha_I_cov'] = np.identity(n)*alpha
gmm_pdf['unit_variance'] = np.identity(n)*alpha

# Draw N_train iid samples of n-dimensional samples of random variable x
X_train_x = create_data_no_labels(
    gmm_pdf['mean_mu'], gmm_pdf['cov_sigma'], N_train)
print(X_train_x.shape)
# Draw N_train iid samples of n-dimensional samples of randomal variable z from 0-mean alpha*I-covariance-matrix
X_train_z = create_data_no_labels(
    gmm_pdf['0_mean'], gmm_pdf['alpha_I_cov'], N_train)
print(X_train_z.shape)
# Draw N_train iid samples of a scalar random variable v from a 0-mean unit-variance Gaussian pdf.
X_train_v = np.random.normal(size=N_train)
# Scalar values of a new random variable


############################# END GENERATE DATA #############################
