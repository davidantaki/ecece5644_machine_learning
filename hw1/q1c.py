from sys import float_info  # Threshold smallest positive floating value
import matplotlib.pyplot as plt  # For general plotting
from matplotlib import cm
import numpy as np
from scipy.stats import multivariate_normal  # MVN not univariate
from sklearn.metrics import confusion_matrix
import prob_utils
from q1a import X_transpose, X, mu, Sigma, sample_labels, estimate_roc, ax_roc,\
    Nl, num_samples, ax_roc, fig_roc, roc_erm, roc_map, prob_error_erm


def perform_lda(X, mu, Sigma, C=2):
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

    mu = np.array([mu[i].reshape(-1, 1) for i in range(C)])
    cov = np.array([Sigma[i].T for i in range(C)])

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


def main():
    # print(X)
    # new_mu = np.mean(X_transpose, axis=(0,1))
    # print("new_mu")
    # print(new_mu)
    # exit()

    # Fisher LDA Classifer (using true model parameters)
    # print(np.transpose(mu))
    # print(Sigma)
    _, discriminant_score_lda = perform_lda(
        X_transpose, np.transpose(mu), Sigma)

    # Estimate the ROC curve for this LDA classifier
    roc_lda, tau_lda = estimate_roc(discriminant_score_lda, sample_labels)
    print(roc_lda)
    print(roc_lda.shape)

    # ROC returns FPR vs TPR, but prob error needs FNR so take 1-TPR
    prob_error_lda = np.array(
        (roc_lda[0, :], 1 - roc_lda[1, :])).T.dot(Nl.T / num_samples)
    print("prob_error_lda:\n{}".format(prob_error_lda))

    # Min prob error
    min_prob_error_lda = np.min(prob_error_lda)
    print("min_prob_error_lda: {}".format(min_prob_error_lda))
    min_ind = np.argmin(prob_error_lda)
    print("min_ind: {}".format(min_ind))

    fig_roc, ax_roc = plt.subplots(figsize=(10, 10))
    ax_roc.plot(roc_erm[0], roc_erm[1])
    ax_roc.plot(roc_map[0], roc_map[1], 'rx', label="Minimum P(Error) MAP", markersize=16)
    ax_roc.set_xlabel(r"Probability of false alarm $P(D=1|L=0)$")
    ax_roc.set_ylabel(r"Probability of correct decision $P(D=1|L=1)$")
    plt.grid(True)

    # Display the estimated ROC curve for LDA and indicate the operating points
    # with smallest empirical error probability estimates (could be multiple)
    ax_roc.plot(roc_lda[0], roc_lda[1], 'b:')
    ax_roc.plot(roc_lda[0, min_ind], roc_lda[1, min_ind],
                'r.', label="Minimum P(Error) LDA", markersize=16)
    ax_roc.set_title("ROC Curves for ERM and LDA")
    ax_roc.legend()

    plt.show()

    # Use min-error threshold
    decisions_lda = discriminant_score_lda >= tau_lda[min_ind]

    # Get indices and probability estimates of the four decision scenarios:
    # (true negative, false positive, false negative, true positive)

    # True Negative Probability
    ind_00_lda = np.argwhere((decisions_lda==0) & (sample_labels==0))
    p_00_lda = len(ind_00_lda) / Nl[0]
    # False Positive Probability
    ind_10_lda = np.argwhere((decisions_lda==1) & (sample_labels==0))
    p_10_lda = len(ind_10_lda) / Nl[0]
    # False Negative Probability
    ind_01_lda = np.argwhere((decisions_lda==0) & (sample_labels==1))
    p_01_lda = len(ind_01_lda) / Nl[1]
    # True Positive Probability
    ind_11_lda = np.argwhere((decisions_lda==1) & (sample_labels==1))
    p_11_lda = len(ind_11_lda) / Nl[1]

    # Display LDA decisions
    fig = plt.figure(figsize=(10, 10))

    # class 0 circle, class 1 +, correct green, incorrect red
    plt.plot(X_transpose[ind_00_lda, 0], X_transpose[ind_00_lda, 1], 'og', label="Correct Class 0")
    plt.plot(X_transpose[ind_10_lda, 0], X_transpose[ind_10_lda, 1], 'or', label="Incorrect Class 0")
    plt.plot(X_transpose[ind_01_lda, 0], X_transpose[ind_01_lda, 1], '+r', label="Incorrect Class 1")
    plt.plot(X_transpose[ind_11_lda, 0], X_transpose[ind_11_lda, 1], '+g', label="Correct Class 1")

    plt.legend()
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title("LDA Decisions (RED incorrect)")
    plt.tight_layout()
    plt.show()

    print("Smallest P(error) for ERM = {}".format(prob_error_erm))
    print("Smallest P(error) for LDA = {}".format(min_prob_error_lda))


if __name__ == '__main__':
    main()
