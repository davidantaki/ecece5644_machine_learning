import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

plt.rc('font', size=22)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=22)   # fontsize of the figure title

n_train = 1000
n_test = 10000


np.random.seed(10)

# GENERATE TRAINING DATASET
theta = np.random.uniform(low=-np.pi, high=np.pi, size=(n_train, 1))
# plt.plot(theta, color="b", marker="s")
# plt.show()
classes = [-1, 1]
r_l = [2, 4]
mean = 0
sigma = 1
n = np.random.multivariate_normal([0, 0], np.identity(2), n_train)
# plt.plot(n, color="b", marker="s")
# plt.show()
X_train = np.empty((n_train, 2))
y_train = np.empty((n_train,))
for i in range(0, n_train):
    # Class -1
    if i < n_train/2:
        X_train[i, 0] = r_l[0]*np.cos(theta[i])+n[i, 0]
        X_train[i, 1] = r_l[0]*np.sin(theta[i])+n[i, 1]
        y_train[i] = -1
    # Class +1
    else:
        X_train[i, 0] = r_l[1]*np.cos(theta[i])+n[i, 0]
        X_train[i, 1] = r_l[1]*np.sin(theta[i])+n[i, 1]
        y_train[i] = 1

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1],
            color="b", marker="o", label="class=-1")
ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color="r",
            marker="o", label="class=+1")
ax.legend()
ax.set_title("Raw Training Dataset")
ax.set_xlabel("x_0")
ax.set_ylabel("x_1")
# plt.show()

# GENERATE TEST DATASET

theta = np.random.uniform(low=-np.pi, high=np.pi, size=(n_test, 1))
# plt.plot(theta, color="b", marker="s")
# plt.show()
classes = [-1, 1]
r_l = [2, 4]
mean = 0
sigma = 1
n = np.random.multivariate_normal([0, 0], np.identity(2), n_test)
# plt.plot(n, color="b", marker="s")
# plt.show()
X_test = np.empty((n_test, 2))
y_test = np.empty((n_test,))
for i in range(0, n_test):
    # Class -1
    if i < n_test/2:
        X_test[i, 0] = r_l[0]*np.cos(theta[i])+n[i, 0]
        X_test[i, 1] = r_l[0]*np.sin(theta[i])+n[i, 1]
        y_test[i] = -1
    # Class +1
    else:
        X_test[i, 0] = r_l[1]*np.cos(theta[i])+n[i, 0]
        X_test[i, 1] = r_l[1]*np.sin(theta[i])+n[i, 1]
        y_test[i] = 1

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(X_test[y_test == -1, 0], X_test[y_test == -1, 1],
            color="b", marker="o", label="class=-1")
ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color="r",
            marker="o", label="class=+1")
ax.legend()
ax.set_title("Raw Testing Dataset")
ax.set_xlabel("x_0")
ax.set_ylabel("x_1")
# plt.show()


svc = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
svc.fit(X_train, y_train)


def plot_svm_predictions(svm):
    # Create coordinate matrices determined by the sample space
    xx, yy = np.meshgrid(np.linspace(-6, 6, 250), np.linspace(-6, 6, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # Matrix of predictions on rid of samples
    y_pred = svm.predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, y_pred, alpha=0.2)


fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(X_train[y_train == classes[0], 0],
         X_train[y_train == classes[0], 1], 'bo', label="Class 0")
ax.plot(X_train[y_train == classes[1], 0],
         X_train[y_train == classes[1], 1], 'ro', label="Class 1")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.legend()
plot_svm_predictions(svc)
plt.show()
