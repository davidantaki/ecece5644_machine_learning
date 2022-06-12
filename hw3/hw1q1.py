# Author: David Antaki

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
from sklearn.model_selection import KFold


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


class TwoLayerMLP(nn.Module):
    # Two-layer MLP (not really a perceptron activation function...) network class

    def __init__(self, input_dim, hidden_dim, C):
        super(TwoLayerMLP, self).__init__()
        # Fully connected layer WX + b mapping from input_dim (n) -> hidden_layer_dim
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        # Output layer again fully connected mapping from hidden_layer_dim -> outputs_dim (C)
        self.output_fc = nn.Linear(hidden_dim, C)
        # Log Softmax (faster and better than straight Softmax)
        # dim=1 refers to the dimension along which the softmax operation is computed
        # In this case computing probabilities across dim 1, i.e., along classes at output layer
        self.log_softmax = nn.LogSoftmax(dim=1)

    # Don't call this function directly!!
    # Simply pass input to model and forward(input) returns output, e.g. model(X)
    def forward(self, X):
        # X = [batch_size, input_dim (n)]
        X = self.input_fc(X)
        # Non-linear activation function, e.g. ReLU (default good choice)
        # Could also choose F.softplus(x) for smooth-ReLU, empirically worse than ReLU
        X = F.relu(X)
        # X = [batch_size, hidden_dim]
        # Connect to last layer and output 'logits'
        X = self.output_fc(X)
        # Squash logits to probabilities that sum up to 1
        y = self.log_softmax(X)
        return y


def model_train(model, data, labels, criterion, optimizer, num_epochs=25):
    # Apparently good practice to set this "flag" too before training
    # Does things like make sure Dropout layers are active, gradients are updated, etc.
    # Probably not a big deal for our toy network, but still worth developing good practice
    model.train()
    # Optimize the neural network
    for epoch in range(num_epochs):
        # These outputs represent the model's predicted probabilities for each class.
        outputs = model(data)
        # Criterion computes the cross entropy loss between input and target
        loss = criterion(outputs, labels)
        # Set gradient buffers to zero explicitly before backprop
        optimizer.zero_grad()
        # Backward pass to compute the gradients through the network
        loss.backward()
        # GD step update
        optimizer.step()

    return model


def model_predict(model, data):
    # Similar idea to model.train(), set a flag to let network know your in "inference" mode
    model.eval()
    # Disabling gradient calculation is useful for inference, only forward pass!!
    with torch.no_grad():
        # Evaluate nn on test data and compare to true labels
        predicted_labels = model(data)
        # Back to numpy
        predicted_labels = predicted_labels.detach().numpy()

        return np.argmax(predicted_labels, 1)

############################ Genarate Data from Gaussian Mixture Model ############################


# Data
gmm_pdf = {}
# Class priors
gmm_pdf['priors'] = np.array([0.5, 0.5, 0.5, 0.5])
# Mean and covariance of data pdfs conditioned on labels
# Gaussian distributions means
gmm_pdf['mean'] = np.array([[2, 2, 2],
                            [-2, -2, -2],
                            [2, -2, 2],
                            [-2, 2, -2]])
# Gaussian distributions covariance matrices
gmm_pdf['cov'] = np.array([[[1, -0.5, 0.3],
                            [-0.5, 1, -0.5],
                            [0.3, -0.5, 1]],
                           [[1, 0.3, -0.2],
                            [0.3, 1, 0.3],
                            [-0.2, 0.3, 1]],
                           [[0.1, 0.4, 0.6],
                            [0.1, 0.3, 0.3],
                            [0.2, 0.1, 0.9]],
                           [[0.8, 0.3, 0.3],
                            [0.1, 0.123, 0.8],
                            [0.7, 0.6, 0.4]]])


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
# Number of samples per component for the training dataset
N = 100
# Number of samples for the test dataset
N_test = 100000
# Total number of training samples
tot_N = N*C
X_train, y_train = create_data(N)
X_test, y_test = create_data(N_test)


fig = plt.figure(figsize=(10, 10))
ax_raw = fig.add_subplot(111, projection='3d')
ax_raw.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
               X_train[y_train == 0, 2], c='r', label="Class 0")
ax_raw.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
               X_train[y_train == 1, 2], c='b', label="Class 1")
ax_raw.scatter(X_train[y_train == 2, 0], X_train[y_train ==
               2, 1], X_train[y_train == 2, 2], 'r*', label="Class 2")
ax_raw.scatter(X_train[y_train == 3, 0], X_train[y_train ==
               3, 1], X_train[y_train == 3, 2], 'g^', label="Class 3")
ax_raw.set_xlabel(r"$x_0$")
ax_raw.set_ylabel(r"$x_1$")
ax_raw.set_zlabel(r"$x_2$")
plt.title("Training Dataset")
plt.legend()
plt.show()

fig = plt.figure(figsize=(10, 10))
ax_raw = fig.add_subplot(111, projection='3d')
ax_raw.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
               X_test[y_test == 0, 2], c='r', label="Class 0")
ax_raw.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
               X_test[y_test == 1, 2], c='b', label="Class 1")
ax_raw.scatter(X_test[y_test == 2, 0], X_test[y_test == 2, 1],
               X_test[y_test == 2, 2], 'r*', label="Class 2")
ax_raw.scatter(X_test[y_test == 3, 0], X_test[y_test == 3, 1],
               X_test[y_test == 3, 2], 'g^', label="Class 3")
ax_raw.set_xlabel(r"$x_0$")
ax_raw.set_ylabel(r"$x_1$")
ax_raw.set_zlabel(r"$x_2$")
plt.title("Testing Dataset")
plt.legend()
# plt.show()

############################ END Genarate Data from Gaussian Mixture Model ############################


############################ Theoretically Optimal Classifier ############################
def get_theoretically_optimal_classifier():
    # If 0-1 loss then yield MAP decision rule, else ERM classifier
    Lambda = np.ones((C, C)) - np.eye(C)

    # ERM decision rule, take index/label associated with minimum conditional risk as decision (N, 1)
    decisions = perform_erm_classification(X_train, Lambda, gmm_pdf, C)

    # Simply using sklearn confusion matrix
    print("Confusion Matrix For Theoretically Optimal Classifier (rows: Predicted class, columns: True class):")
    conf_mat = confusion_matrix(decisions, y_train)
    print(conf_mat)

    correct_class_samples = np.sum(np.diag(conf_mat))
    print("Total Mumber of Misclassified Samples: {:d}".format(
        tot_N - correct_class_samples))

    prob_error = 1 - (correct_class_samples / tot_N)
    print(
        "Empirically Estimated Probability of Error: {:.4f}".format(prob_error))

############################ END Theoretically Optimal Classifier ############################

################## Model Order Selection using Cross Validation ##################
# Here we perform Cross Validation to decide on the best number of perceptrons to use in our MLP.


def get_model_order_with_cross_validation():
    # Range of perceptrons we will try
    n_perceptrons = np.arange(1, 30, 1)
    max_n_perceptrons = np.max(n_perceptrons)

    # Number of folds for CV
    K = 10

    # STEP 1: Partition the dataset into K approximately-equal-sized partitions
    # Shuffles data before doing the division into folds (not necessary, but a good idea)
    kf = KFold(n_splits=K, shuffle=True)

    # Allocate space for CV
    # No need for training loss storage too but useful comparison
    mse_valid_mk = np.empty((max_n_perceptrons, K))
    # Indexed by model m, data partition k
    mse_train_mk = np.empty((max_n_perceptrons, K))

    # STEP 2: Try all polynomial orders between 1 (best line fit) and 21 (big time overfit) M=2
    for n in n_perceptrons:
        # K-fold cross validation
        k = 0
        # NOTE that these subsets are of the TRAINING dataset
        # Imagine we don't have enough data available to afford another entirely separate validation set
        for train_indices, valid_indices in kf.split(X_train):
            # Extract the training and validation sets from the K-fold split
            X_train_k = X_train[train_indices]
            y_train_k = y_train[train_indices]
            X_valid_k = X_train[valid_indices]
            y_valid_k = y_train[valid_indices]

            input_dim = X_train_k.shape[1]
            # n_hidden_neurons = 16
            output_dim = C
            model = TwoLayerMLP(input_dim, n, output_dim)
            # Visualize network architecture
            print(model)

            # Stochastic GD with learning rate and momentum hyperparameters
            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.01, momentum=0.9)
            # The nn.CrossEntropyLoss() loss function automatically performs a log_softmax() to
            # the output when validating, on top of calculating the negative log-likelihood using
            # nn.NLLLoss(), while also being more stable numerically... So don't implement from scratch
            criterion = nn.CrossEntropyLoss()
            num_epochs = 100

            # Convert numpy structures to PyTorch tensors, as these are the data types required by the library
            X_train_k_tensor = torch.FloatTensor(X_train_k)
            y_train_k_tensor = torch.LongTensor(y_train_k)
            X_valid_k_tensor = torch.FloatTensor(X_valid_k)
            y_valid_k_tensor = torch.LongTensor(y_valid_k)

            # Train the model
            model = model_train(model, X_train_k_tensor, y_train_k_tensor, criterion,
                                optimizer, num_epochs=num_epochs)

            # Using the trained model get the predicted classifications/labels from the test and validation sets.
            y_train_k_pred = model_predict(model, X_train_k_tensor)
            y_valid_k_pred = model_predict(model, X_valid_k_tensor)

            # Get the classification error probability for the training dataset.
            y_train_conf_mat = confusion_matrix(y_train_k, y_train_k_pred)
            print(y_train_conf_mat)
            y_train_correct_class_samples = np.sum(np.diag(y_train_conf_mat))
            y_train_correct_tot_N = np.sum(y_train_conf_mat)
            print("y_train_correct_class_samples Total Mumber of Misclassified Samples: {:d}".format(
                y_train_correct_tot_N - y_train_correct_class_samples))

            y_train_prob_error = 1 - (y_train_correct_class_samples /
                                      y_train_correct_tot_N)
            print(
                "y_train_correct_class_samples Empirically Estimated Probability of Error: {:.4f}".format(y_train_prob_error))

            # Get the classification error probability for the validation dataset.
            y_valid_conf_mat = confusion_matrix(y_valid_k, y_valid_k_pred)
            print(y_valid_conf_mat)
            y_valid_correct_class_samples = np.sum(np.diag(y_valid_conf_mat))
            y_valid_correct_tot_N = np.sum(y_valid_conf_mat)
            print("y_valid_correct_class_samples Total Mumber of Misclassified Samples: {:d}".format(
                y_valid_correct_tot_N - y_valid_correct_class_samples))

            y_valid_prob_error = 1 - (y_valid_correct_class_samples /
                                      y_valid_correct_tot_N)
            print(
                "y_valid_correct_class_samples Empirically Estimated Probability of Error: {:.4f}".format(y_valid_prob_error))

            # Record classification error probabilities as well for this model and k-fold
            mse_train_mk[n - 1, k] = y_train_prob_error
            mse_valid_mk[n - 1, k] = y_valid_prob_error

            # Increment to the next k-fold
            k += 1

            #############################################################
            # # Train model parameters
            # poly_train_features_cv = PolynomialFeatures(
            #     degree=deg, include_bias=True)
            # X_train_k_poly = poly_train_features_cv.fit_transform(X_train_k)
            # theta_mk = mle_solution(X_train_k_poly, y_train_k)

            # # Validation fold polynomial transformation
            # X_valid_k_poly = poly_train_features_cv.transform(X_valid_k)

            # # Make predictions on both the training and validation set
            # y_train_k_pred = X_train_k_poly.dot(theta_mk)
            # y_valid_k_pred = X_valid_k_poly.dot(theta_mk)

            # # Record MSE as well for this model and k-fold
            # mse_train_mk[deg - 1, k] = mse(y_train_k_pred, y_train_k)
            # mse_valid_mk[deg - 1, k] = mse(y_valid_k_pred, y_valid_k)
            # k += 1
            #############################################################

    # STEP 3: Compute the average probability of error for that model (based in this case on number of perceptrons
    # Model average CV loss over folds
    mse_train_m = np.mean(mse_train_mk, axis=1)
    mse_valid_m = np.mean(mse_valid_mk, axis=1)

    print(mse_train_m)
    print(mse_valid_m)

    # +1 as the index starts from 0 while the degrees start from 1
    optimal_d = np.argmin(mse_valid_m) + 1
    print("The model selected to best fit the data without overfitting is: d={}".format(optimal_d))

# STEP 4: Re-train using your optimally selected model (degree=3) and deploy!!
# ...


'''
# Create coordinate matrices determined by the sample space
xx, yy, zz = np.meshgrid(np.linspace(-4, 4, 250),
                         np.linspace(-4, 4, 250), np.linspace(-4, 4, 250))

grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

# Set up test data as tensor
grid_tensor = torch.FloatTensor(grid)
# Z matrix are the predictions resulting from the forward pass through the network
Z = model_predict(model, grid_tensor).reshape(xx.shape)
print(Z.shape)

plt.figure(figsize=(10, 8))

# uses gray background for black dots
# plt.pcolormesh(xx, yy, Z[:,:], cmap=plt.cm.coolwarm)

fig = plt.figure(figsize=(10, 10))
ax_raw = fig.add_subplot(111, projection='3d')
ax_raw.scatter(X[:, 0], X[:, 1], X[:, 2], c='r', label="Class 0")
# ax_raw.scatter(Z[y == 1, 0], Z[y == 1, 1],
#                Z[y == 1, 2], c='b', label="Class 1")
# ax_raw.scatter(Z[y == 2, 0], Z[y == 2, 1], Z[y == 2, 2], 'r*', label="Class 2")
# ax_raw.scatter(Z[y == 3, 0], Z[y == 3, 1], Z[y == 3, 2], 'g^', label="Class 3")
ax_raw.set_xlabel(r"$x_0$")
ax_raw.set_ylabel(r"$x_1$")
ax_raw.set_zlabel(r"$x_2$")
plt.show()

# plt.plot(X[y==0, 0], X[y==0, 1], 'bx', label="Class 0")
# plt.plot(X[y==1, 0], X[y==1, 1], 'ko', label="Class 1")
# plt.plot(X[y==2, 0], X[y==2, 1], 'r*', label="Class 2")
# plt.xlabel(r"$x_1$")
# plt.ylabel(r"$x_2$")
# plt.title("MLP Classification Boundaries Train Set")
# plt.legend()
# plt.show()
'''


get_theoretically_optimal_classifier()
exit()
get_model_order_with_cross_validation()
