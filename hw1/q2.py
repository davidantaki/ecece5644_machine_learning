from sys import float_info  # Threshold smallest positive floating value
import matplotlib.pyplot as plt  # For general plotting
from matplotlib import cm
import numpy as np
from scipy.stats import multivariate_normal  # MVN not univariate
from sklearn.metrics import confusion_matrix
from modules import prob_utils

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
mu = np.array([[1, 1],
               [3, 3],
               [5, 5],
               [7, 7]])
mu_transpose = np.transpose(mu)

# Cov. matrices to be scaled versions of the identity matrix (with scale
# factors that lead to a significant amount of overlap between the data
# from these Gaussians)
identity = np.identity(2)
print(identity)
Sigma = np.zeros((4, 2, 2))
print(Sigma)
Sigma[0] = identity * 10.01
Sigma[1] = identity * 10.11
Sigma[2] = identity * 10.31
Sigma[3] = identity * 9.92
print(Sigma)


# Determine dimensionality from mixture PDF parameters
dimensions = mu.shape[1]
print("dimensions: {}".format(dimensions))

# Create PDF parameter structure
gmm_params = prob_utils.GaussianMixturePDFParameters(
    priors, num_classes, mu_transpose, np.transpose(Sigma))
gmm_params.print_pdf_params()

# print(gmm_params.component_pdfs[0].mean.shape)

# Output samples and labels
X = np.zeros([num_samples, dimensions])
sample_labels = np.zeros(num_samples)
# print("labels: {}".format(sample_labels))


# Generate 3D matrix from a mixture of 3 Gaussians
X, sample_labels = prob_utils.generate_mixture_samples(
    num_samples, dimensions, gmm_params, True)


# Actual number of samples generated from each class
Nl = np.array([sum(sample_labels == l) for l in class_labels])
print("Number of samples from Class 0: {:d}, Class 1: {:d}, Class 2: {:d},Class 3: {:d},".format(
    Nl[0], Nl[1], Nl[2], Nl[3]))

# # Get true data distribution knowledge
# class0_mu = np.zeros((2))
# for i in range(0,10000):
#     if sample_labels[i] == 0:
#         class0_mu[0] = class0_mu[0] + X[0,i]
#         class0_mu[1] = class0_mu[1] + X[1,i]

# class0_mu[0] = class0_mu[0] / Nl[0]
# class0_mu[1] = class0_mu[1] / Nl[1]
# print(class0_mu)

# class0_np_mu = np.mean(np.transpose(X), 0)
# print(class0_np_mu)
# # print(class0_sum/Nl[0])
# exit()

C = num_classes


# MAP classifier (is a special case of ERM corresponding to 0-1 loss)
# 0-1 loss values yield MAP decision rule
Lambda = np.ones((C, C)) - np.identity(C)
print("Loss Matrix:\n{}".format(Lambda))

# Min prob. of error classifier
# Conditional likelihoods of each class given x, shape (C, N)
# Row 0 = likelihoods that given the sample, it is of Class 0.
# Row 1 = likelihoods that given the sample, it is of Class 1.
class_cond_likelihoods = np.array(
    [multivariate_normal.pdf(np.transpose(X), mu[c], Sigma[c]) for c in range(C)])
print("class_cond_likelihoods:\n{}".format(class_cond_likelihoods))
class_priors = np.diag(priors)
print(class_cond_likelihoods.shape)
print(class_priors.shape)
class_posteriors = class_priors.dot(class_cond_likelihoods)
print(class_posteriors)

# We want to create the risk matrix of size 4 x N
cond_risk = Lambda.dot(class_posteriors)
print(cond_risk)

# Get the decision for each column in risk_mat
decisions = np.argmin(cond_risk, axis=0)
print(decisions.shape)

# Plot for decisions vs true labels
fig = plt.figure(figsize=(12, 10))
marker_shapes = '.o^s'  # Accomodates up to C=5
marker_colors = 'brgmy'

# Get sample class counts
sample_class_counts = np.array([sum(sample_labels == j) for j in class_labels])
X_transpose = np.transpose(X)
# Confusion matrix
conf_mat = np.zeros((C, C))
for i in class_labels:  # Each decision option
    for j in class_labels:  # Each class label
        ind_ij = np.argwhere((decisions == i) & (sample_labels == j))
        # Average over class sample count
        conf_mat[i, j] = len(ind_ij)/sample_class_counts[j]

        # True label = Marker shape; Decision = Marker Color
        if i == j:
            marker = marker_shapes[j] + 'g'
            label = "Correct Class "+str(i)
            plt.plot(X_transpose[ind_ij, 0], X_transpose[ind_ij,
                     1], marker, markersize=3, label=label)
        elif i != j:
            marker = marker_shapes[j] + 'r'
            label = "Incorrect Class "+str(i)
            plt.plot(X_transpose[ind_ij, 0], X_transpose[ind_ij,
                     1], marker, markersize=3, label=label)

print("Confusion matrix:")
print(conf_mat)

print("Minimum Probability of Error:")
prob_error = 1 - np.diag(conf_mat).dot(sample_class_counts / num_samples)
print(prob_error)

plt.legend(loc=2, prop={'size': 10})
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.tight_layout()
plt.title(
    "Minimum Probability of Error Classified Sampled Data:  {:.3f}".format(prob_error))
plt.show()


##################### PART B #####################

# New Loss matrix
Lambda = np.zeros((C, C))
Lambda[0][0] = 0
Lambda[0][1] = 1
Lambda[0][2] = 2
Lambda[0][3] = 3
Lambda[1][0] = 1
Lambda[1][1] = 0
Lambda[1][2] = 1
Lambda[1][3] = 2
Lambda[2][0] = 2
Lambda[2][1] = 1
Lambda[2][2] = 0
Lambda[2][3] = 1
Lambda[3][0] = 3
Lambda[3][1] = 2
Lambda[3][2] = 1
Lambda[3][3] = 0
print("Loss Matrix:\n{}".format(Lambda))

# Min prob. of error classifier
# Conditional likelihoods of each class given x, shape (C, N)
# Row 0 = likelihoods that given the sample, it is of Class 0.
# Row 1 = likelihoods that given the sample, it is of Class 1.
class_cond_likelihoods = np.array(
    [multivariate_normal.pdf(np.transpose(X), mu[c], Sigma[c]) for c in range(C)])
print("class_cond_likelihoods:\n{}".format(class_cond_likelihoods))
class_priors = np.diag(priors)
print(class_cond_likelihoods.shape)
print(class_priors.shape)
class_posteriors = class_priors.dot(class_cond_likelihoods)
print(class_posteriors)

# We want to create the risk matrix of size 4 x N
cond_risk = Lambda.dot(class_posteriors)
print(cond_risk)

# Get the decision for each column in risk_mat
decisions = np.argmin(cond_risk, axis=0)
print(decisions.shape)

# Plot for decisions vs true labels
fig = plt.figure(figsize=(12, 10))
marker_shapes = '.o^s'  # Accomodates up to C=5
marker_colors = 'brgmy'

# Get sample class counts
sample_class_counts = np.array([sum(sample_labels == j) for j in class_labels])
X_transpose = np.transpose(X)
# Confusion matrix
conf_mat = np.zeros((C, C))
for i in class_labels:  # Each decision option
    for j in class_labels:  # Each class label
        ind_ij = np.argwhere((decisions == i) & (sample_labels == j))
        # Average over class sample count
        conf_mat[i, j] = len(ind_ij)/sample_class_counts[j]

        # True label = Marker shape; Decision = Marker Color
        if i == j:
            marker = marker_shapes[j] + 'g'
            label = "Correct Class "+str(i)
            plt.plot(X_transpose[ind_ij, 0], X_transpose[ind_ij,
                     1], marker, markersize=3, label=label)
        elif i != j:
            marker = marker_shapes[j] + 'r'
            label = "Incorrect Class "+str(i)
            plt.plot(X_transpose[ind_ij, 0], X_transpose[ind_ij,
                     1], marker, markersize=3, label=label)

print("Confusion matrix:")
print(conf_mat)

print("Minimum Probability of Error:")
prob_error = 1 - np.diag(conf_mat).dot(sample_class_counts / num_samples)
print(prob_error)

plt.legend(loc=2, prop={'size': 10})
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.tight_layout()
plt.title(
    "Minimum Probability of Error Classified Sampled Data:  {:.3f}".format(prob_error))
plt.show()
