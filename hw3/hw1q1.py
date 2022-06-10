# Widget to manipulate plots in Jupyter notebooks
from torchsummary import summary
import torch.nn.functional as F
import torch.nn as nn
import torch
from sklearn.linear_model import LogisticRegression
from scipy.stats import multivariate_normal as mvn
import numpy as np
import matplotlib.colors as mcol
import matplotlib.pyplot as plt  # For general plotting


# Utility to visualize PyTorch network and shapes

N = 100


def create_data(N, noise=0.1):
    # Uses the same covariance matrix, scaled identity, for all Gaussians
    Sigma = noise * np.eye(2)
    # Five gaussian means specified to span a square and its centre
    Gs = [
        mvn(mean=[2, 2], cov=Sigma),
        mvn(mean=[-2, -2], cov=Sigma),
        mvn(mean=[2, -2], cov=Sigma),
        mvn(mean=[-2, 2], cov=Sigma),
        mvn(mean=[0, 0], cov=Sigma),
    ]
    # Draw random variable samples and assign labels, note class 3 has less samples altogether
    X = np.concatenate([G.rvs(size=N) for G in Gs])
    y = np.concatenate((np.zeros(N), np.zeros(
        N), np.ones(N), np.ones(N), 2 * np.ones(N)))

    # Will return an X and y of shapes (5*N, 2) and (5*N)
    # Representing our dataset of 2D samples
    return X, y


X, y = create_data(N)
C = len(np.unique(y))

plt.figure(figsize=(10, 8))
plt.plot(X[y == 0, 0], X[y == 0, 1], 'bx', label="Class 0")
plt.plot(X[y == 1, 0], X[y == 1, 1], 'ko', label="Class 1")
plt.plot(X[y == 2, 0], X[y == 2, 1], 'r*', label="Class 2")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Data and True Labels")
plt.legend()
plt.show()
