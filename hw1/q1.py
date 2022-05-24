import matplotlib.pyplot as plt  # For general plotting
from matplotlib import cm
import numpy as np
from scipy.stats import multivariate_normal  # MVN not univariate
import prob_utils

np.set_printoptions(suppress=True)

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)

plt.rc('font', size=22)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('figure', titlesize=22)  # fontsize of the figure title

# Number of samples to draw from each distribution
num_samples = 10000

# Likelihood of each distribution to be selected AND class priors!!!
priors = np.array([0.65, 0.35])

# Determine number of classes/mixture components
num_classes = len(priors)
print("num_classes: {}".format(num_classes))

# Gaussian distributions means
mu = np.array([[-0.5, -0.5, -0.5],
               [1, 1, 1]])

# Gaussian distributions covariance matrices
Sigma = np.array([[[1, -0.5, 0.3],
                   [-0.5, 1, -0.5],
                   [0.3, -0.5, 1]],
                  [[1, 0.3, -0.2],
                   [0.3, 1, 0.3],
                   [-0.2, 0.3, 1]]])

# Determine dimensionality from mixture PDF parameters
dimensions = mu.shape[1]
print("dimensions: {}".format(dimensions))

# Create PDF parameter structure
gmm_params = prob_utils.GaussianMixturePDFParameters(
    priors, num_classes, mu, np.transpose(Sigma))
gmm_params.print_pdf_params()

# Generate 3D matrix from a mixture of 3 Gaussians
_, _ = prob_utils.generate_mixture_samples(num_samples, dimensions, gmm_params, True)

# # Output samples and labels
# X = np.zeros([num_samples, dimensions])
# sample_labels = np.zeros(num_samples)
# print("labels: {}".format(sample_labels))

# possible_labels = np.array([0, 1])

# # Decide randomly which samples will come from each component
# u = np.random.rand(num_samples)
# thresholds = np.cumsum(priors)
# print("thresholds: {}".format(thresholds))

# for c in range(num_classes):
#     # Get randomly sampled indices for this component
#     c_ind = np.argwhere(u <= thresholds[c])[:, 0]
#     c_N = len(c_ind)  # No. of samples in this component
#     sample_labels[c_ind] = c * np.ones(c_N)
#     # Multiply by 1.1 to fail <= thresholds and thus not reuse samples
#     u[c_ind] = 1.1 * np.ones(c_N)
#     X[c_ind, :] = multivariate_normal.rvs(mu[c], Sigma[c], c_N)

# # Plot the original data and their true labels
# plt.figure(figsize=(12, 10))
# plt.plot(X[sample_labels == 0, 0],
#          X[sample_labels == 0, 1], 'bo', label="Class 0")
# plt.plot(X[sample_labels == 1, 0],
#          X[sample_labels == 1, 1], 'rx', label="Class 1")
# plt.legend()
# plt.xlabel(r"$x_0$")
# plt.ylabel(r"$x_1$")
# plt.title("Data and True Labels")
# plt.tight_layout()
# plt.show()

# Actual number of samples generated from each class
# Nl = np.array([sum(sample_labels == l) for l in possible_labels])
# print("Number of samples from Class 0: {:d}, Class 1: {:d}".format(
#     Nl[0], Nl[1]))
