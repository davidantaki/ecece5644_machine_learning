from random import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


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

# GENERATE TRAINING DATASET
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
        c = np.random.random_integers(0,1)
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


svc = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma=1.0, C=1.0))
svc.fit(X_train, y_train)


def plot_svm_predictions(svm):
    # Create coordinate matrices determined by the sample space
    xx, yy = np.meshgrid(np.linspace(-8, 8, 250), np.linspace(-8, 8, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # Matrix of predictions on rid of samples
    y_pred = svm.predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, y_pred, alpha=0.2)


fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(X_train[y_train == classes[0], 0],
         X_train[y_train == classes[0], 1], 'bo', label="Class -1")
ax.plot(X_train[y_train == classes[1], 0],
         X_train[y_train == classes[1], 1], 'ro', label="Class +1")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.legend()
plot_svm_predictions(svc)
plt.show()
