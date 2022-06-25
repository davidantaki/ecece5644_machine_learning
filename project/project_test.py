from enum import unique
import random
from scipy.stats import multivariate_normal  # MVN not univariate
import matplotlib.pyplot as plt  # For general plotting
import pandas
import numpy as np
from sklearn.metrics import confusion_matrix

# Get sample labels aka "activity labels". These are the POSSIBLE sample labels but not the labels associated with data.
dataset = pandas.read_csv(
    '..\\ecece5644_machine_learning\\project\\songs_normalize.csv')
print(dataset)
print(dataset.shape)
dataset = np.array(dataset)
unique_artists = []
for sample in dataset:
    if sample[0] not in unique_artists:
        unique_artists.append(sample[0])

print(unique_artists)
print(len(unique_artists))



# possible_sample_labels = np.array(possible_sample_labels)
# print("possible_sample_labels:\n{}".format(possible_sample_labels))
# print("possible_sample_labels.shape:\n{}".format(possible_sample_labels.shape))

# # Column headers/features of the samples.
# col_names = pandas.read_csv(
#     'UCI HAR Dataset\\UCI HAR Dataset\\features.txt', sep=' ', names=['num', 'feature'])
# col_names = np.array(col_names)
# num_features = len(col_names)
# print("col_names:\n{}".format(col_names))
# print(col_names)
# print(col_names[:, 1])
# print(col_names[:, 1].shape)
# print("num_features:\n{}".format(num_features))
