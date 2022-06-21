from datetime import datetime
from random import random
from typing import final
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


plt.rc('font', size=22)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=22)   # fontsize of the figure title

n_train = 1000
n_test = 10000
classes = [-1, 1]


np.random.seed(10)


def generate_data(n_samples: np.integer):
    theta = np.random.uniform(low=-np.pi, high=np.pi, size=(n_samples, 1))
    # plt.plot(theta, color="b", marker="s")
    # plt.show()
    r_l = [2, 4]
    n = np.random.multivariate_normal([0, 0], np.identity(2), n_samples)
    # plt.plot(n, color="b", marker="s")
    # plt.show()
    X = np.empty((n_samples, 2))
    y = np.empty((n_samples,))
    for i in range(0, n_samples):
        c = np.random.random_integers(0, 1)
        # Class -1
        if c == 0:
            X[i, 0] = r_l[0]*np.cos(theta[i])+n[i, 0]
            X[i, 1] = r_l[0]*np.sin(theta[i])+n[i, 1]
            y[i] = -1
        # Class +1
        else:
            X[i, 0] = r_l[1]*np.cos(theta[i])+n[i, 0]
            X[i, 1] = r_l[1]*np.sin(theta[i])+n[i, 1]
            y[i] = 1
    return X, y


# GENERATE TRAINING DATASET
X_train, y_train = generate_data(n_train)
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1],
#             color="b", marker="o", label="class=-1")
# ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color="r",
#             marker="o", label="class=+1")
# ax.legend()
# ax.set_title("Raw Training Dataset")
# ax.set_xlabel("x_0")
# ax.set_ylabel("x_1")
# plt.show()

# GENERATE TEST DATASET
X_test, y_test = generate_data(n_test)
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.scatter(X_test[y_test == -1, 0], X_test[y_test == -1, 1],
#             color="b", marker="o", label="class=-1")
# ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color="r",
#             marker="o", label="class=+1")
# ax.legend()
# ax.set_title("Raw Testing Dataset")
# ax.set_xlabel("x_0")
# ax.set_ylabel("x_1")
# plt.show()


def plot_svm_predictions(svm):
    # Create coordinate matrices determined by the sample space
    xx, yy = np.meshgrid(np.linspace(-8, 8, 250), np.linspace(-8, 8, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # Matrix of predictions on rid of samples
    y_pred = svm.predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, y_pred, alpha=0.2)


# Number of C and gamma hyperparams tried
n_C_list = 10
n_gamma_list = 10
n_param_total = n_C_list*n_gamma_list
# These are the possible hyper parameters for C and gamma
C_list = np.logspace(-3, 3, num=n_C_list, base=10.0)
gamma_list = np.logspace(-3, 3, num=n_gamma_list, base=10.0)

'''
C_list = [0.1, 1.0, 10, 100]
gamma_list = [0.1, 1.0, 100]
'''

# For storing scores from cross validation
# This has the shape (number of total hyperparam combos tried, 3)
# 3=(C, gamma, scores)
final_scores = []

iteration_counter = 0
for C_param in C_list:
    for gamma_param in gamma_list:
        # https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-pysvc = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma=1.0, C=1.0))
        # ^^ Suggests gamma and C range of 10^-3 to 10^3 to start
        svc = make_pipeline(StandardScaler(), SVC(
            kernel='rbf', gamma=gamma_param, C=C_param))
        # svc.fit(X_train, y_train)
        # 'accuracy' scorer in sklearn uses the number of misclassified samples divided by the total number of samples, therefore giving the minimum probability of classification error.
        scores = cross_validate(estimator=svc, X=X_train,
                                y=y_train, cv=10, scoring='accuracy')
        final_scores.append([C_param, gamma_param, scores])
        print("Itararion: {}\tC_param: {}\tgamma_param: {}\tProb. Error: {}".format(iteration_counter,
                                                                                    C_param, gamma_param, 1-np.mean(scores['test_score'])))
        ''''
        # SVC with poly degree features
        # Pipeline of sequentially applied transforms before producing the final estimation, e.g. Support Vector Classifier
        svc.fit(X_train, y_train)

        plt.figure(figsize=(10, 8))
        plt.plot(X_train[y_train == -1, 0],
                 X_train[y_train == -1, 1], 'bx', label="Class -1")
        plt.plot(X_train[y_train == 1, 0],
                 X_train[y_train == 1, 1], 'ko', label="Class +1")
        plt.xlabel(r"$x_0$")
        plt.ylabel(r"$x_1$")
        plt.title("C={}, gamma={}, P(error)={}".format(
            C_param, gamma_param, 1-np.mean(scores['test_score'])))
        plt.legend()
        plot_svm_predictions(svc)
        # plt.show()
        plt.savefig("{}-C={}, gamma={}.png".format(
            datetime.now().strftime("%Y-%d-%m-%H-%M"), C_param, gamma_param), dpi=300)
        '''
        iteration_counter = iteration_counter + 1

# print(final_scores)
final_scores = np.array(final_scores)
print(final_scores)
# Average all the scores and get rid of the other metrics from cross_validate.
# And convert to prob. of error from accuracy.
for scores in final_scores:
    scores[2] = 1-np.mean(scores[2]['test_score'])
print("final_scores: {}".format(final_scores))
# print(final_scores[:, 2])
optimal_params = final_scores[np.argmin(final_scores[:, 2])]
print("optimal_params: {}".format(optimal_params))
print("Optimal C: {}\tOptimal gamma: {}\tProb_error: {}".format(
    optimal_params[0], optimal_params[1], optimal_params[2]))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(final_scores[:, 0], final_scores[:, 1],
           final_scores[:, 2], c='r')
ax.set_xlabel("C")
ax.set_ylabel("Gamma")
ax.set_zlabel("P(error)")
ax.set_title(
    "Box Constraint (C) vs Gaussian Kernel Width (gamma) vs P(error) for that C and gamma combo")
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(final_scores[:, 0], final_scores[:, 2],
           color="b", marker="o")
ax.legend()
ax.set_xscale('log')
ax.set_title("C vs P(error)")
ax.set_xlabel("C")
ax.set_ylabel("P(error)")
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(final_scores[:, 1], final_scores[:, 2],
           color="b", marker="o")
ax.legend()
ax.set_xscale('log')
ax.set_title("Gamma vs P(error)")
ax.set_xlabel("gamma")
ax.set_ylabel("P(error)")
plt.show()
