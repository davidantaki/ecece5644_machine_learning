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


N = 100
sigma0 = [[1, -0.5, 0.3],
          [-0.5, 1, -0.5],
          [0.3, -0.5, 1]]
sigma1 = [[1, 0.3, -0.2],
          [0.3, 1, 0.3],
          [-0.2, 0.3, 1]]
sigma2 = [[1, -0.5, 0.3],
          [-0.5, 1, -0.5],
          [0.3, -0.5, 1]]
sigma3 = [[1, 0.3, -0.2],
          [0.3, 1, 0.3],
          [-0.2, 0.3, 1]]


def create_data(N):
    # 4 classes
    Gs = [
        mvn(mean=[2, 2, 2], cov=sigma0),
        mvn(mean=[-2, -2, -2], cov=sigma1),
        mvn(mean=[2, -2, 2], cov=sigma2),
        mvn(mean=[-2, 2, -2], cov=sigma3)
    ]

    # Draw random variable samples and assign labels, note class 3 has less samples altogether
    # G = each guassian distribution
    X = np.concatenate([G.rvs(size=N) for G in Gs])
    y = np.concatenate((np.zeros(N), np.ones(
        N), 2*np.ones(N), 3*np.ones(N)))

    # Will return an X of shape (4*N, 3dimensions)
    # 4 classes each with N samples ^^
    # y of shape (4*N) -> 1 label per sample
    return X, y


X, y = create_data(N)
# Number of classes
C = 4

# Plot 2 features of the generated data.
plt.figure(figsize=(10, 8))
plt.plot(X[y == 0, 0], X[y == 0, 1], 'bx', label="Class 0")
plt.plot(X[y == 1, 0], X[y == 1, 1], 'ko', label="Class 1")
plt.plot(X[y == 2, 0], X[y == 2, 1], 'r*', label="Class 2")
plt.plot(X[y == 3, 0], X[y == 3, 1], 'g^', label="Class 3")
plt.xlabel(r"$x_0$")
plt.ylabel(r"$x_1$")
plt.title("Data and True Labels")
plt.legend()
plt.show()