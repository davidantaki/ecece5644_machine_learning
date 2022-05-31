import random
from scipy.stats import multivariate_normal  # MVN not univariate
import matplotlib.pyplot as plt  # For general plotting
import pandas
import numpy as np
from sklearn.metrics import confusion_matrix

# Get sample labels aka "activity labels". These are the POSSIBLE sample labels but not the labels associated with data.
possible_sample_labels = pandas.read_csv(
    'UCI HAR Dataset\\UCI HAR Dataset\\activity_labels.txt', sep=' ', names=['acivity_num', 'activity_title'])
possible_sample_labels = np.array(possible_sample_labels)
print("possible_sample_labels:\n{}".format(possible_sample_labels))
print("possible_sample_labels.shape:\n{}".format(possible_sample_labels.shape))

# Column headers/features of the samples.
col_names = pandas.read_csv(
    'UCI HAR Dataset\\UCI HAR Dataset\\features.txt', sep=' ', names=['num', 'feature'])
col_names = np.array(col_names)
num_features = len(col_names)
print("col_names:\n{}".format(col_names))
print(col_names)
print(col_names[:, 1])
print(col_names[:, 1].shape)
print("num_features:\n{}".format(num_features))


# Get Data
all_activity_data = pandas.read_csv(
    'UCI HAR Dataset\\UCI HAR Dataset\\train\\X_train.csv', sep=',', names=None)
all_activity_data = np.array(all_activity_data)
X = np.array(all_activity_data)
X_transpose = np.transpose(X)
# print("all_activity_data:\n{}".format(all_activity_data))
# print(all_activity_data.shape)

# Get sample labels associated with the data.
sample_labels = pandas.read_csv(
    'UCI HAR Dataset\\UCI HAR Dataset\\train\\y_train.txt', sep=' ', names=None)
sample_labels = np.reshape(np.array(sample_labels), (1, 7351))
sample_labels = np.array(sample_labels[0, :])
print("sample_labels:\n{}".format(sample_labels))
print("len(sample_labels):\n{}".format(len(sample_labels)))

# print("new_sample_labels:\n{}".format(new_sample_labels))
# print("len(new_sample_labels):\n{}".format(len(new_sample_labels)))

# Number of samples
num_samples = len(X)
print("num_samples:\n{}".format(num_samples))

# Get class parameters
used_classes, class_counts = np.unique(
    sample_labels, return_counts=True)
print("num_samples:\n{}".format(num_samples))
print(used_classes)
num_classes = len(used_classes)
print("num_classes:\n{}".format(num_classes))
priors = class_counts/num_samples
print("priors:\n{}".format(priors))
class_means = np.array(
    [np.mean(X[np.argwhere(sample_labels == c)], axis=0) for c in used_classes])
# print("class_means:\n{}".format(class_means))
class_cov = np.array([np.cov(X[sample_labels == c].T)
                      for c in used_classes])
# print("class_cov:\n{}".format(class_cov))

# Regularization
eigvals, eigvects = np.linalg.eig(np.cov(X.T))
reg_param = np.mean(eigvals)

mu = class_means.reshape(num_classes, num_features)
Sigma = class_cov + reg_param*np.identity(num_features)
class_cond_likelihoods = np.array(
    [multivariate_normal.pdf(X, mu[c], Sigma[c]) for c in range(num_classes)])
class_priors = np.diag(priors)
class_posteriors = class_priors.dot(class_cond_likelihoods)

decisions = np.argmax(class_posteriors, axis=0) + \
    np.min(used_classes) * np.ones(num_samples)

conf_matrix = confusion_matrix(decisions, sample_labels)
print("conf_matrix:\n{}".format(conf_matrix))

prob_errors = len(np.argwhere(decisions != sample_labels))
print('# of Errors', prob_errors, "\nEst. P(error)", prob_errors/num_samples)

# Plot all data
# print(X.shape)
# print(X)
# plt.plot(X)
# plt.show()

# Plots random subsets of the data in 3-dimensions.
# Does not perform PCA nor plot PCA data.
visualize_raw_data_3d = False
if visualize_raw_data_3d:
    for i in range(0, 5):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_ind = random.randint(0, num_features)
        y_ind = random.randint(0, num_features)
        z_ind = random.randint(0, num_features)
        while(y_ind == x_ind):
            y_ind = random.randint(0, num_features)
        while(z_ind == x_ind or z_ind == y_ind):
            z_ind = random.randint(0, num_features)
        ax.scatter(X_transpose[x_ind, :],
                   X_transpose[y_ind, :], X_transpose[z_ind, :])
        ax.set_xlabel(col_names[:, 1][x_ind])
        ax.set_ylabel(col_names[:, 1][y_ind])
        ax.set_zlabel(col_names[:, 1][z_ind])
        ax.set_title("Acitivity Data Visualization")
        plt.autoscale()
        plt.show()

# This will visualize subsets of features randomly.
# It will plot the raw feature data, and then run PCA on the data and plot the PCA data.
visualize_raw_data_2d = True
if visualize_raw_data_2d:
    for i in range(0, 5):
        raw_data_fig = plt.figure(num=1, figsize=(10, 10))
        x_ind = random.randint(0, num_features)
        y_ind = random.randint(0, num_features)
        while(y_ind == x_ind):
            y_ind = random.randint(0, num_features)
        plt.scatter(X[:, x_ind], X[:, y_ind])
        plt.xlabel(col_names[:, 1][x_ind])
        plt.ylabel(col_names[:, 1][y_ind])
        plt.title("Activity Raw Data Visulization")
        plt.autoscale()
        raw_data_fig.show()

        ######## Do PCA ########

        # First derive sample-based estimates of mean vector and covariance matrix:
        mu_hat = np.mean(X, axis=0)
        Sigma_hat = np.cov(X.T)

        # Mean-subtraction is a necessary assumption for PCA, so perform this to obtain zero-mean sample set
        C = X - mu_hat

        # Get the eigenvectors (in U) and eigenvalues (in D) of the estimated covariance matrix
        lambdas, U = np.linalg.eig(Sigma_hat)
        # Get the indices from sorting lambdas in order of increasing value, with ::-1 slicing to then reverse order
        idx = lambdas.argsort()[::-1]
        # Extract corresponding sorted eigenvectors and eigenvalues
        U = U[:, idx]
        D = np.diag(lambdas[idx])

        # Calculate the PC projections of zero-mean samples (in z)
        Z = C.dot(U)

        # Let's see what it looks like only along the first two PCs
        pca_fig = plt.figure(num=2, figsize=(10, 10))
        plt.scatter(Z[:, x_ind], Z[:, y_ind])
        plt.xlabel(col_names[:, 1][x_ind])
        plt.ylabel(col_names[:, 1][y_ind])
        plt.title("Activity Data PCA Projections to 2D Space")
        plt.show()
