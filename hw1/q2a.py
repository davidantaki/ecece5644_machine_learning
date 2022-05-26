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
Sigma[0] = identity * 1.01
Sigma[1] = identity * 1.11
Sigma[2] = identity * 1.31
Sigma[3] = identity * 0.92
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
    [multivariate_normal.pdf(np.transpose(X), mu2[c], Sigma[c]) for c in range(C)])
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

exit()
discriminant_score_erm = np.log(
    class_cond_likelihoods[1]) - np.log(class_cond_likelihoods[0])

# Gamma threshold for MAP decision rule (remove Lambdas and you obtain same gamma on priors only; 0-1 loss simplification)
gamma_map = (Lambda[1, 0] - Lambda[0, 0]) / \
    (Lambda[0, 1] - Lambda[1, 1]) * priors[0]/priors[1]
# Same as:
# gamma_map = priors[0]/priors[1]
print(gamma_map)

decisions_map = discriminant_score_erm >= np.log(gamma_map)

# Get indices and probability estimates of the four decision scenarios:
# (true negative, false positive, false negative, true positive)

# True Negative Probability
ind_00_map = np.argwhere((decisions_map == 0) & (sample_labels == 0))
p_00_map = len(ind_00_map) / Nl[0]
# False Positive Probability
ind_10_map = np.argwhere((decisions_map == 1) & (sample_labels == 0))
p_10_map = len(ind_10_map) / Nl[0]
# False Negative Probability
ind_01_map = np.argwhere((decisions_map == 0) & (sample_labels == 1))
p_01_map = len(ind_01_map) / Nl[1]
# True Positive Probability
ind_11_map = np.argwhere((decisions_map == 1) & (sample_labels == 1))
p_11_map = len(ind_11_map) / Nl[1]

# Probability of error for MAP classifier, empirically estimated
prob_error_erm = np.array((p_10_map, p_01_map)).dot(Nl.T / num_samples)
# print(np.array((p_10_map, p_01_map)).shape)
# Display MAP decisions
fig = plt.figure(figsize=(10, 10))

# class 0 circle, class 1 +, correct green, incorrect red
# print(X.shape)
# print(ind_00_map)
X_transpose = np.transpose(X)
plt.plot(X_transpose[ind_00_map, 0],
         X_transpose[ind_00_map, 1], 'og', label="Correct Class 0")
plt.plot(X_transpose[ind_10_map, 0],
         X_transpose[ind_10_map, 1], 'or', label="Incorrect Class 0")
plt.plot(X_transpose[ind_01_map, 0],
         X_transpose[ind_01_map, 1], '+r', label="Incorrect Class 1")
plt.plot(X_transpose[ind_11_map, 0],
         X_transpose[ind_11_map, 1], '+g', label="Correct Class 1")

plt.legend()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("MAP Decisions (RED incorrect)")
plt.tight_layout()
plt.show()


# Generate ROC curve samples

def estimate_roc(discriminant_score, label):
    Nlabels = np.array((sum(label == 0), sum(label == 1)))

    sorted_score = sorted(discriminant_score)

    # Use tau values that will account for every possible classification split
    taus = ([sorted_score[0] - float_info.epsilon] +
            sorted_score +
            [sorted_score[-1] + float_info.epsilon])

    # Calculate the decision label for each observation for each gamma
    decisions = [discriminant_score >= t for t in taus]

    ind10 = [np.argwhere((d == 1) & (label == 0)) for d in decisions]
    p10 = [len(inds)/Nlabels[0] for inds in ind10]
    ind11 = [np.argwhere((d == 1) & (label == 1)) for d in decisions]
    p11 = [len(inds)/Nlabels[1] for inds in ind11]

    # ROC has FPR on the x-axis and TPR on the y-axis
    roc = np.array((p10, p11))

    return roc, taus


# Construct the ROC for ERM by changing log(gamma)
roc_erm, _ = estimate_roc(discriminant_score_erm, sample_labels)
roc_map = np.array((p_10_map, p_11_map))

fig_roc, ax_roc = plt.subplots(figsize=(10, 10))
ax_roc.plot(roc_erm[0], roc_erm[1])
ax_roc.plot(roc_map[0], roc_map[1], 'rx',
            label="Minimum P(Error) MAP", markersize=16)
ax_roc.legend()
ax_roc.set_xlabel(r"Probability of false alarm $P(D=1|L=0)$")
ax_roc.set_ylabel(r"Probability of correct decision $P(D=1|L=1)$")

print(roc_map)
print(roc_erm)

plt.grid(True)
plt.show()


# Take diag so we have (C, C) shape of priors with prior prob along diagonal
class_priors = np.diag(priors)
print("class_priors:\n{}".format(class_priors))
# class_priors*likelihood with diagonal matrix creates a matrix of posterior probabilities
# with each class as a row and N columns for samples, e.g. row 1: [p(y1)p(x1|y1), ..., p(y1)p(xN|y1)]
class_posteriors = class_priors.dot(class_cond_likelihoods)

# MAP rule, take largest class posterior per example as your decisions matrix (N, 1)
# Careful of indexing! Added np.ones(N) just for difference in starting from 0 in Python and labels={1,2,3}
decisions = np.argmax(class_posteriors, axis=0)

# Simply using sklearn confusion matrix
print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(decisions, sample_labels)
print(conf_mat)
correct_class_samples = np.sum(np.diag(conf_mat))
print("Total Number of Misclassified Samples: {:d}".format(
    num_samples - correct_class_samples))

# Alternatively work out probability error based on incorrect decisions per class
# perror_per_class = np.array(((conf_mat[1,0]+conf_mat[2,0])/Nl[0], (conf_mat[0,1]+conf_mat[2,1])/Nl[1], (conf_mat[0,2]+conf_mat[1,2])/Nl[2]))
# prob_error = perror_per_class.dot(Nl.T / N)

prob_error = 1 - (correct_class_samples / num_samples)
print("Empirically Estimated Probability of Error: {:.4f}".format(prob_error))

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
