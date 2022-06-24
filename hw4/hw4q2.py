import argparse
from sklearn.model_selection import KFold, cross_val_score
from msilib.schema import Component
import matplotlib.pyplot as plt

import numpy as np
import pandas

from skimage.io import imread
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_predict

np.set_printoptions(suppress=True)

np.random.seed(7)

plt.rc('font', size=22)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=22)   # fontsize of the figure title


def generate_feature_vector(image):
    # Load image, get its dimensions
    image_np = np.array(image)
    # print(image_np)
    # Return an array of the row and column indices of the image (height and width)
    img_indices = np.indices((image_np.shape[0], image_np.shape[1]))
    # print(img_indices)

    # Set up data array to store features: row ind, col ind, [num_channels]
    # num_channels = 1 for grayscale and 3 for RGB
    if image_np.ndim == 2:  # Grayscale image
        # Create the features matrix of row and col indices, plus pixel values
        features = np.array(
            [img_indices[0].flatten(), img_indices[1].flatten(), image_np.flatten()])
        # Find ranges of features as max - min
        min_f = np.min(features, axis=1)
        max_f = np.max(features, axis=1)
        ranges = max_f - min_f
        # Each feature normalized to the unit interval [0,1] using max-min normalization: (x - min) / (max - min)
        # New axis to allow numpy broadcasting
        # np.diag(1/ranges) to perform the division operation in matrix form
        normalized_data = np.diag(
            1 / ranges).dot(features - min_f[:, np.newaxis])
    elif image_np.ndim == 3:  # Color image with RGB values
        # Create the features matrix of row and col indices, plus pixel values
        features = np.array([img_indices[0].flatten(), img_indices[1].flatten(),
                             image_np[..., 0].flatten(), image_np[..., 1].flatten(), image_np[..., 2].flatten()])
        min_f = np.min(features, axis=1)
        max_f = np.max(features, axis=1)
        ranges = max_f - min_f
        # Each feature normalized to the unit interval [0,1] using max-min normalization: (x - min) / (max - min)
        # New axis np.newaxis to allow numpy broadcasting
        # np.diag(1/ranges) to perform the division operation in matrix form
        normalized_data = np.diag(
            1 / ranges).dot(features - min_f[:, np.newaxis])
    else:
        print("Incorrect image dimensions for feature vector")

    # Returns feature vector of normalized pixels as shape (height*width, 3 or 5)
    return image_np, normalized_data.T


# Load image
paraskier_color = imread(
    'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/html/images/plain/normal/color/60079.jpg')

# Show image
# fig1, ax1 = plt.subplots(1, 1, figsize=(10, 10))
# ax1.imshow(paraskier_color)
# ax1.set_title("ParaSkier Color")
# plt.show()

# Get image feature vector=(row_index, col_index, R, G, B)
paraskier_np, paraskier_norm = generate_feature_vector(paraskier_color)
print(paraskier_norm.shape)

# Visualize the Components
# fig2 = plt.figure(figsize=(10, 10))
# ax2 = fig2.add_subplot(111, projection='3d')
# ax2.scatter(paraskier_norm[:, 2], paraskier_norm[:, 3],
#             paraskier_norm[:, 4], c='r')
# ax2.set_xlabel("Red Component")
# ax2.set_ylabel("Green Component")
# ax2.set_zlabel("Blue Component")
# ax2.set_title(
#     "Visualizing the Image's RGB Components")
# plt.show()

# fig3, ax3 = plt.subplots(1, 1, figsize=(10, 10))
# ax3.scatter(paraskier_norm[:, 2], paraskier_norm[:, 3])
# ax3.set_title("ParaSkier Color")
# ax3.set_xlabel("Red Component")
# ax3.set_ylabel("Green Component")
# plt.show()

cv_scores = []
iteration_counter = 0
# for n_components in range(1, 2):
#     gmm = GaussianMixture(n_components=n_components, covariance_type="full")
#     gmm.fit(paraskier_norm)
#     gmm.predict_proba()
#     gmm.score()
#     scores = cross_val_predict(estimator=gmm, X=paraskier_norm,
#                                cv=10)
#     print(scores)
#     cv_scores.append([n_components, scores])
#     print("Itararion: {}\tn_components: {}".format(iteration_counter,
#                                                     n_components))
#     iteration_counter = iteration_counter + 1


def graph_log_likelihood_vs_n_components():
    scores = []
    for n_components in range(1, 100):
        gmm = GaussianMixture(n_components=n_components,
                              covariance_type="full")
        gmm.fit(paraskier_norm)
        s = gmm.score(paraskier_norm)
        print(s)
        scores.append([n_components, s])

    print(scores)
    scores = np.array(scores)
    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 10))
    ax4.set_ylabel("Log Likelihood")
    ax4.set_xlabel("# Components")
    ax4.set_title("Log Likelihood vs # Components")
    ax4.plot(scores[:, 0], scores[:, 1])
    plt.show()


def get_optimal_number_of_gmm_components():

    # Cross Validation Params
    K = 2
    max_num_components_to_try = 10

    # Partition data
    kf = KFold(n_splits=K, shuffle=True)
    n_components_list = np.arange(1, max_num_components_to_try)

    # Allocate space for CV
    scores = []
    scores = np.empty((max(n_components_list), K))

    # CROSS VALIDATION
    for n_components in n_components_list:
        # K-fold cross validation
        k = 0
        # NOTE that these subsets are of the TRAINING dataset
        for train_indices, valid_indices in kf.split(paraskier_norm):
            # Extract the training and validation sets from the K-fold split
            X_train_k = paraskier_norm[train_indices]
            X_valid_k = paraskier_norm[valid_indices]

            # Fit GMM on training data
            gmm = GaussianMixture(n_components=n_components,
                                  covariance_type="full")
            gmm.fit(X_train_k)

            # Get log-likelihood on the validation set
            s = gmm.score(X_valid_k)
            scores[n_components-1, k] = s
            k += 1

            print(
                "n_components: {}\tk-fold: {}\tLog-Likelihood: {}".format(n_components, k, s))

    print(scores)
    # Compute Average of scores
    scores = np.mean(scores, axis=1)
    print(scores)

    # Choose optimal number of components.
    # Choose the number of components that maximizes the log-likelikhood.
    # Stop optimizing when the log-likelihood stops increasing by more than 0.2
    opt_threshold = 0.2
    opt_s_i = -1
    for i in range(1, len(scores)):
        if scores[i]-scores[i-1] < opt_threshold:
            opt_s_i = i-1
            break


    # Plot Cross Validation results
    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 10))
    ax4.set_ylabel("Log Likelihood")
    ax4.set_xlabel("# Components")
    ax4.set_title("Log Likelihood vs # Components")
    ax4.plot(n_components_list, scores)
    ax4.plot(opt_s_i+1, scores[opt_s_i], marker='x', c='r')
    plt.show()

    opt_n_component = opt_s_i+1
    print(opt_n_component)
    return opt_n_component

# def train_final_model(n_components):
#     # TRAIN FINAL MODEL

#     # PRINT SEGMENTED IMAGE
#     gmm = GaussianMixture(n_components=opt_n_component,
#                         covariance_type="full")
#     gmm.fit(paraskier_norm)
#     pixel_labels = gmm.predict(paraskier_norm)
#     # print(np.unique(pixel_labels))
#     # counts = pandas.Series(pixel_labels).value_counts()
#     # for i in np.unique(pixel_labels):
#     #     print(counts.get(i))

#     fig4, ax4 = plt.subplots(1, 1, figsize=(10, 10))
#     ax4.set_title("ParaSkier Segmented with {}-Components".format(2))
#     labels_img = pixel_labels.reshape(paraskier_np.shape[0], paraskier_np.shape[1])
#     ax4.imshow(labels_img)
#     plt.show()
#     '''

#     '''
#     # Create figure to plot all K-means segmentation results for each example image
#     fig, ax = plt.subplots(4, 4, figsize=(15, 15))
#     ax[0, 0].imshow(bird_gray, cmap=plt.cm.gray)
#     ax[0, 0].set_title("Bird Grey")
#     ax[0, 0].set_axis_off()
#     ax[1, 0].imshow(bird_color)
#     ax[1, 0].set_title("Bird Color")
#     ax[1, 0].set_axis_off()
#     ax[2, 0].imshow(plane_gray, cmap=plt.cm.gray)
#     ax[2, 0].set_title("Plane Grey")
#     ax[2, 0].set_axis_off()
#     ax[3, 0].imshow(plane_color)
#     ax[3, 0].set_title("Plane Color")
#     ax[3, 0].set_axis_off()

### K-means Algorithm ###

plt.show()


# graph_log_likelihood_vs_n_components()
get_optimal_number_of_gmm_components()
