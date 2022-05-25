import matplotlib.pyplot as plt  # For general plotting
from matplotlib import cm
import numpy as np
from scipy.stats import multivariate_normal  # MVN not univariate
from sklearn.metrics import confusion_matrix
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
priors = np.array([0.2, 0.25, 0.25, 0.3])

# Determine number of classes/mixture components
num_classes = len(priors)
print("num_classes: {}".format(num_classes))

class_labels = np.array([0, 1, 2, 3])

# Mean vectors approx. equally spaces out along a line
mu = np.array([[0, 0],
               [1, 1],
               [2, 2],
               [3, 3]])

# Cov. matrices to be scaled versions of the identity matrix (with scale
# factors that lead to a significant amount of overlap between the data
# from these Gaussians)
identity = np.identity(2)
print(identity)
Sigma = np.zeros((4,2,2))
print(Sigma)
Sigma[0] = identity * 1
Sigma[1] = identity * 2
Sigma[2] = identity * 3
Sigma[3] = identity * 4
print(Sigma)

'''
# Determine dimensionality from mixture PDF parameters
dimensions = mu.shape[1]
print("dimensions: {}".format(dimensions))

# Create PDF parameter structure
gmm_params = prob_utils.GaussianMixturePDFParameters(
    priors, num_classes, mu, np.transpose(Sigma))
gmm_params.print_pdf_params()

# Output samples and labels
X = np.zeros([num_samples, dimensions])
sample_labels = np.zeros(num_samples)
print("labels: {}".format(sample_labels))

# Generate 3D matrix from a mixture of 3 Gaussians
X, sample_labels = prob_utils.generate_mixture_samples(
    num_samples, dimensions, gmm_params, True)


possible_labels = np.array([0, 1])

# Actual number of samples generated from each class
Nl = np.array([sum(sample_labels == l) for l in possible_labels])
print("Number of samples from Class 0: {:d}, Class 1: {:d}".format(
    Nl[0], Nl[1]))

C = num_classes

# Min prob. of error classifier
# Conditional likelihoods of each class given x, shape (C, N)
print(mu2[0].shape)
class_cond_likelihoods = np.array(
    [multivariate_normal.pdf(X, mu2[c], Sigma[c]) for c in range(C)])
print(class_cond_likelihoods)
# Take diag so we have (C, C) shape of priors with prior prob along diagonal
class_priors = np.diag(priors)
# class_priors*likelihood with diagonal matrix creates a matrix of posterior probabilities
# with each class as a row and N columns for samples, e.g. row 1: [p(y1)p(x1|y1), ..., p(y1)p(xN|y1)]
class_posteriors = class_priors.dot(class_cond_likelihoods)

# MAP rule, take largest class posterior per example as your decisions matrix (N, 1)
# Careful of indexing! Added np.ones(N) just for difference in starting from 0 in Python and labels={1,2,3}
decisions = np.argmax(class_posteriors, axis=0) + np.ones(num_samples)

# Simply using sklearn confusion matrix
print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(decisions, sample_labels)
print(conf_mat)
correct_class_samples = np.sum(np.diag(conf_mat))
print("Total Mumber of Misclassified Samples: {:d}".format(
    num_samples - correct_class_samples))

# Alternatively work out probability error based on incorrect decisions per class
# perror_per_class = np.array(((conf_mat[1,0]+conf_mat[2,0])/Nl[0], (conf_mat[0,1]+conf_mat[2,1])/Nl[1], (conf_mat[0,2]+conf_mat[1,2])/Nl[2]))
# prob_error = perror_per_class.dot(Nl.T / N)

prob_error = 1 - (correct_class_samples / num_samples)
print("Empirically Estimated Probability of Error: {:.4f}".format(prob_error))
'''