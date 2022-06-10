# Widget to manipulate plots in Jupyter notebooks
from torchsummary import summary
import torch.nn.functional as F
import torch.nn as nn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from scipy.stats import multivariate_normal as mvn
import numpy as np
import matplotlib.colors as mcol
import matplotlib.pyplot as plt  # For general plotting
from sys import float_info  # Threshold smallest positive floating value


def perform_lda(X, labels, C=2):
    """  Fisher's Linear Discriminant Analysis (LDA) on data from two classes (C=2).

    In practice the mean and covariance parameters would be estimated from training samples.

    Args:
        X: Real-valued matrix of samples with shape [N, n], N for sample count and n for dimensionality.
        mu: Mean vector [C, n].
        Sigma: Covariance matrices [C, n, n].

    Returns:
        w: Fisher's LDA project vector, shape [n, 1].
        z: Scalar LDA projections of input samples, shape [N, 1].
    """

    # First, estimate the class-conditional pdf mean and covariance matrices from samples
    # Note that reshape ensures my return mean vectors are of 2D shape (column vectors nx1)
    mu = np.array([np.mean(X[labels == i], axis=0).reshape(-1, 1)
                  for i in range(C)])
    cov = np.array([np.cov(X[labels == i].T) for i in range(C)])

    # Determine between class and within class scatter matrix
    Sb = (mu[1] - mu[0]).dot((mu[1] - mu[0]).T)
    Sw = cov[0] + cov[1]

    # Regular eigenvector problem for matrix Sw^-1 Sb
    lambdas, U = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    # Get the indices from sorting lambdas in order of increasing value, with ::-1 slicing to then reverse order
    idx = lambdas.argsort()[::-1]

    # Extract corresponding sorted eigenvectors
    U = U[:, idx]

    # First eigenvector is now associated with the maximum eigenvalue, mean it is our LDA solution weight vector
    w = U[:, 0]

    # Scalar LDA projections in matrix form
    z = X.dot(w)

    return w, z


# ERM classification rule (min prob. of error classifier)
def perform_erm_classification(X, Lambda, gmm_params, C):
    # Conditional likelihoods of each x given each class, shape (C, N)
    class_cond_likelihoods = np.array(
        [mvn.pdf(X, gmm_params['mean'][c], gmm_params['cov'][c]) for c in range(C)])

    # Take diag so we have (C, C) shape of priors with prior prob along diagonal
    class_priors = np.diag(gmm_params['priors'])
    # class_priors*likelihood with diagonal matrix creates a matrix of posterior probabilities
    # with each class as a row and N columns for samples, e.g. row 1: [p(y1)p(x1|y1), ..., p(y1)p(xN|y1)]
    class_posteriors = class_priors.dot(class_cond_likelihoods)

    # Conditional risk matrix of size C x N with each class as a row and N columns for samples
    risk_mat = Lambda.dot(class_posteriors)

    return np.argmin(risk_mat, axis=0)

# Generate ROC curve samples


def estimate_roc(discriminant_score, labels):
    N_labels = np.array((sum(labels == 0), sum(labels == 1)))

    # Sorting necessary so the resulting FPR and TPR axes plot threshold probabilities in order as a line
    sorted_score = sorted(discriminant_score)

    # Use gamma values that will account for every possible classification split
    gammas = ([sorted_score[0] - float_info.epsilon] +
              sorted_score +
              [sorted_score[-1] + float_info.epsilon])

    # Calculate the decision label for each observation for each gamma
    decisions = [discriminant_score >= g for g in gammas]

    # Retrieve indices where FPs occur
    ind10 = [np.argwhere((d == 1) & (labels == 0)) for d in decisions]
    # Compute FP rates (FPR) as a fraction of total samples in the negative class
    p10 = [len(inds) / N_labels[0] for inds in ind10]
    # Retrieve indices where TPs occur
    ind11 = [np.argwhere((d == 1) & (labels == 1)) for d in decisions]
    # Compute TP rates (TPR) as a fraction of total samples in the positive class
    p11 = [len(inds) / N_labels[1] for inds in ind11]

    # ROC has FPR on the x-axis and TPR on the y-axis, but return others as well for convenience
    roc = {}
    roc['p10'] = np.array(p10)
    roc['p11'] = np.array(p11)

    return roc, gammas


def get_binary_classification_metrics(predictions, labels):
    N_labels = np.array((sum(labels == 0), sum(labels == 1)))

    # Get indices and probability estimates of the four decision scenarios:
    # (true negative, false positive, false negative, true positive)
    class_metrics = {}

    # True Negative Probability Rate
    ind_00 = np.argwhere((predictions == 0) & (labels == 0))
    class_metrics['tnr'] = len(ind_00) / N_labels[0]
    # False Positive Probability Rate
    ind_10 = np.argwhere((predictions == 1) & (labels == 0))
    class_metrics['fpr'] = len(ind_10) / N_labels[0]
    # False Negative Probability Rate
    ind_01 = np.argwhere((predictions == 0) & (labels == 1))
    class_metrics['fnr'] = len(ind_01) / N_labels[1]
    # True Positive Probability Rate
    ind_11 = np.argwhere((predictions == 1) & (labels == 1))
    class_metrics['tpr'] = len(ind_11) / N_labels[1]

    return class_metrics


def generate_data_from_gmm(N, pdf_params):
    # Determine dimensionality from mixture PDF parameters
    n = pdf_params['m'].shape[1]
    # Output samples and labels
    X = np.zeros([N, n])
    y = np.zeros(N)

    # Decide randomly which samples will come from each component
    u = np.random.rand(N)
    thresholds = np.cumsum(pdf_params['priors'])
    thresholds = np.insert(thresholds, 0, 0)  # For intervals of classes

    L = np.array(range(1, len(pdf_params['priors'])+1))
    for l in L:
        # Get randomly sampled indices for this component
        indices = np.argwhere(
            (thresholds[l-1] <= u) & (u <= thresholds[l]))[:, 0]
        # No. of samples in this component
        Nl = len(indices)
        y[indices] = l * np.ones(Nl) - 1
        if n == 1:
            X[indices, 0] = norm.rvs(
                pdf_params['m'][l-1], pdf_params['C'][l-1], Nl)
        else:
            X[indices, :] = mvn.rvs(
                pdf_params['m'][l-1], pdf_params['C'][l-1], Nl)

    return X, y


def create_data(N):
    # Draw random variable samples and assign labels, note class 3 has less samples altogether
    # G = each guassian distribution
    X = np.concatenate([G.rvs(size=N) for G in Gs])
    y = np.concatenate((np.zeros(N), np.ones(
        N), 2*np.ones(N), 3*np.ones(N)))

    # Will return an X of shape (4*N, 3dimensions)
    # 4 classes each with N samples ^^
    # y of shape (4*N) -> 1 label per sample
    return X, y


# Data
gmm_pdf = {}
# Class priors
gmm_pdf['priors'] = np.array([0.5, 0.5, 0.5, 0.5])
# Mean and covariance of data pdfs conditioned on labels
gmm_pdf['mean'] = np.array([[2, 2, 2],
                            [-2, -2, -2],
                            [2, -2, 2],
                            [-2, 2, -2]])  # Gaussian distributions means
gmm_pdf['cov'] = np.array([[[1, -0.5, 0.3],
                            [-0.5, 1, -0.5],
                            [0.3, -0.5, 1]],
                           [[1, 0.3, -0.2],
                            [0.3, 1, 0.3],
                            [-0.2, 0.3, 1]],
                           [[1, -0.5, 0.3],
                            [-0.5, 1, -0.5],
                            [0.3, -0.5, 1]],
                           [[1, 0.3, -0.2],
                            [0.3, 1, 0.3],
                            [-0.2, 0.3, 1]]])  # Gaussian distributions covariance matrices


# 4 classes
Gs = [
    mvn(mean=gmm_pdf['mean'][0], cov=gmm_pdf['cov'][0]),
    mvn(mean=gmm_pdf['mean'][1], cov=gmm_pdf['cov'][1]),
    mvn(mean=gmm_pdf['mean'][2], cov=gmm_pdf['cov'][2]),
    mvn(mean=gmm_pdf['mean'][3], cov=gmm_pdf['cov'][3])
]

# Number of classes
C = 4
# Possible Labels
L = np.array(range(C))
# Number of samples per component
N = 100
# Total number of samples
tot_N = N*C
X, y = create_data(N)


fig = plt.figure(figsize=(10, 10))
ax_raw = fig.add_subplot(111, projection='3d')
ax_raw.scatter(X[y == 0, 0], X[y == 0, 1],
               X[y == 0, 2], c='r', label="Class 0")
ax_raw.scatter(X[y == 1, 0], X[y == 1, 1],
               X[y == 1, 2], c='b', label="Class 1")
ax_raw.scatter(X[y == 2, 0], X[y == 2, 1], X[y == 2, 2], 'r*', label="Class 2")
ax_raw.scatter(X[y == 3, 0], X[y == 3, 1], X[y == 3, 2], 'g^', label="Class 3")
ax_raw.set_xlabel(r"$x_0$")
ax_raw.set_ylabel(r"$x_1$")
ax_raw.set_zlabel(r"$x_2$")
plt.show()

############################ Theoretically Optimal Classifier ############################

# If 0-1 loss then yield MAP decision rule, else ERM classifier
Lambda = np.ones((C, C)) - np.eye(C)

# ERM decision rule, take index/label associated with minimum conditional risk as decision (N, 1)
decisions = perform_erm_classification(X, Lambda, gmm_pdf, C)

# Simply using sklearn confusion matrix
print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(decisions, y)
print(conf_mat)

correct_class_samples = np.sum(np.diag(conf_mat))
print("Total Mumber of Misclassified Samples: {:d}".format(
    tot_N - correct_class_samples))

prob_error = 1 - (correct_class_samples / tot_N)
print("Empirically Estimated Probability of Error: {:.4f}".format(prob_error))
