import random
from scipy.stats import multivariate_normal  # MVN not univariate
import matplotlib.pyplot as plt  # For general plotting
import pandas
import numpy as np
from sklearn.metrics import confusion_matrix

print("hello")

# Get Data
all_wine_data = pandas.read_csv('winequality-white.csv', sep=';')
print("all_wine_data:\n{}".format(all_wine_data))
print(all_wine_data.shape)

col_labels = all_wine_data.columns
print("col_labels:\n{}".format(col_labels))
# Sample labels is the "quality" column
sample_labels = np.array(all_wine_data[all_wine_data.columns[11]])
print(sample_labels.shape)

print("labels:\n{}".format(sample_labels))
X = np.array(all_wine_data[all_wine_data.columns[0:11]])
print("samples:\n{}".format(X))
num_samples = len(sample_labels)
print("num_samples:\n{}".format(num_samples))
print(X.shape)

actual_class_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# Get class parameters
used_classes, class_counts = np.unique(sample_labels, return_counts=True)
print("num_samples:\n{}".format(num_samples))
print(used_classes)
num_classes = len(used_classes)
print("num_classes:\n{}".format(num_classes))
priors = class_counts/num_samples
print("priors:\n{}".format(priors))
class_means = np.array(
    [np.mean(X[np.argwhere(sample_labels == c)], axis=0) for c in used_classes])
# print("class_means:\n{}".format(class_means))
class_cov = np.array([np.cov(X[sample_labels == c].T) for c in used_classes])
# print("class_cov:\n{}".format(class_cov))

# Regularization
eigvals, eigvects = np.linalg.eig(np.cov(X.T))
reg_param = np.mean(eigvals)

mu = class_means.reshape(num_classes, 11)
Sigma = class_cov + reg_param*np.identity(11)
class_cond_likelihoods = np.array(
    [multivariate_normal.pdf(X, mu[c], Sigma[c]) for c in range(num_classes)])
class_priors = np.diag(priors)
class_posteriors = class_priors.dot(class_cond_likelihoods)

decisions = np.argmax(class_posteriors, axis=0) + \
    np.min(used_classes) * np.ones(num_samples)

conf_matrix = confusion_matrix(decisions, sample_labels)
print(conf_matrix)

prob_errors = len(np.argwhere(decisions != sample_labels))
print('# of Errors', prob_errors, "\nEst. P(error)", prob_errors/num_samples)

# # MAP classifier (is a special case of ERM corresponding to 0-1 loss)
# # 0-1 loss values yield MAP decision rule
# Lambda = np.ones((num_classes, num_classes)) - np.identity(num_classes)
# print("Loss Matrix:\n{}".format(Lambda))

# # We want to create the risk matrix of size 4 x N
# cond_risk = Lambda.dot(class_posteriors)
# print("cond_risk:\n{}".format(cond_risk))

# # Get the decision for each column in risk_mat
# decisions = np.argmin(cond_risk, axis=0)
# # print(decisions.shape)

# # Get sample class counts
# sample_class_counts = class_counts
# # sample_class_counts = np.array([sum(sample_labels == j) for j in actual_class_labels])
X_transpose = np.transpose(X)
# # Confusion matrix
# conf_mat = np.zeros((num_classes, num_classes))

# print("Confusion matrix:")
# print(conf_mat)

# print("Minimum Probability of Error:")
# prob_error = 1 - np.diag(conf_matrix).dot(class_counts / num_samples)
# print(prob_error)


# Output samples and labels
# x = np.zeros([n, N])
# y = np.zeros(N)

# Decide randomly which samples will come from each component
# u = np.random.rand(N)
# thresholds = np.cumsum(pdf_params.priors)

# for c in range(pdf_params.C):
#     # Get randomly sampled indices for this component
#     c_ind = np.argwhere(u <= thresholds[c])[:, 0]
#     c_N = len(c_ind)  # No. of samples in this component
#     y[c_ind] = c * np.ones(c_N)
#     # Multiply by 1.1 to fail <= thresholds and thus not reuse samples
#     u[c_ind] = 1.1 * np.ones(c_N)
#     x[:, c_ind] = generate_random_samples(
#         c_N, n, pdf_params.component_pdfs[c], visualize=False)

for i in range(0, 5):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_ind = random.randint(0, 10)
    y_ind = random.randint(0, 10)
    z_ind = random.randint(0, 10)
    while(y_ind == x_ind):
        y_ind = random.randint(0, 10)
    while(z_ind == x_ind or z_ind == y_ind):
        z_ind = random.randint(0, 10)
    ax.scatter(X_transpose[x_ind, :],
               X_transpose[y_ind, :], X_transpose[z_ind, :])
    ax.set_xlabel(col_labels[x_ind])
    ax.set_ylabel(col_labels[y_ind])
    ax.set_zlabel(col_labels[z_ind])
    ax.set_title("Wine Data Visulization")
    plt.autoscale()
    plt.show()
