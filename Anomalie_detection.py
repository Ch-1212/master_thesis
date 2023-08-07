import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import adjusted_rand_score
from sktime.distances import pairwise_distance
import util_plots as uplot

own_daily_label = pd.read_csv(r'data/own_labels_daily.csv')

def z_score_standardization_matrix(series):
    column_means = np.nanmean(series, axis=0)
    column_stds = np.nanstd(series, axis=0)

    # Normalize each column using the calculated means and standard deviations
    return (series - column_means) / column_stds


# Anomalie 1: Unusal pattern of valid clusters
# Detect "normal" week, give alarm when this "normal" pattern is disturbed
# Only needs labels as input (daily_labels would be enough)
# Use ARI (to be free of permutations)
def detect_anomalie_1(data, daily_labels):
    pattern = own_daily_label[14:21]['own_labels'] #one week from Fr-Fr
    anomalie_indexes = []
    for i in range(0, len(daily_labels) - 7, 7):
        label_week = daily_labels[i:i+7]
        ari = adjusted_rand_score(pattern, label_week)
        if ari != 1.0:
            anomalie_indexes.append(data.index[i*96]) #i)
            #print("Unsual pattern of valid clusters in week ", i)
    return anomalie_indexes


# Anomalie 2: Non-valid operating state
# Detect how similar data point is to cluster center
# Needs cluster centers, plus original data in window shape (or redo windows)
def detect_anomalie_2(data, distances):

    # Keep only the lowest value for each row (as I am only interested in the real cluster that it was assigned to)
    min_distances = np.full_like(distances, np.nan)
    # Find the indices of the minimum distances for each row
    min_indices = np.argmin(distances, axis=1)
    # Assign the minimum distances to the corresponding positions in the min_distances array
    min_distances[np.arange(distances.shape[0]), min_indices] = distances[np.arange(distances.shape[0]), min_indices]
    min_distances_z = z_score_standardization_matrix(min_distances)

    count_non_nan = np.count_nonzero(~np.isnan(min_distances), axis=0) #84, 74, 70
    max_values = np.nanmax(min_distances_z, axis=0) #5.3, 3.4, 3.0
    min_values = np.nanmin(min_distances_z, axis=0)  # -1.1, -1.2, -1.4

    def plot_distances_per_cluster():
        plt.figure()
        plt.rcParams.update({'font.size': 16})
        plt.plot(np.sort(min_distances_z[:, 0]))
        plt.plot(np.sort(min_distances_z[:, 1]))
        plt.plot(np.sort(min_distances_z[:, 2]))
        # plt.title('Distances')
        plt.xlabel('Data points')
        plt.ylabel('Distances per cluster')
        plt.show()

    #plot_distances_per_cluster()

    # Flatten the array and filter out NaN values
    min_distances_z_flat = np.ravel(min_distances_z)[~np.isnan(np.ravel(min_distances_z))]

    # Calculate the threshold for 5% of the data to become anomalies
    threshold = np.percentile(min_distances_z_flat, 95)

    def plot_distances():
        plt.figure()
        plt.rcParams.update({'font.size': 16})
        plt.plot(np.sort(min_distances_z_flat))
        # plt.title('Distances')
        plt.xlabel('Data points')
        plt.ylabel('Distances')
        plt.show()

    #plot_distances()

    anomalie_indexes = []
    for i in range(len(min_distances_z_flat)):
        if min_distances_z_flat[i] >= threshold:
            anomalie_indexes.append(data.index[i*96])
            #print("Distance for data point is higher than threshold:", min_distances_z_flat[i])

    return anomalie_indexes


def detect_ts_anomalie(data, x_, kmeans):
    distances = pairwise_distance(x_, kmeans.cluster_centers_, metric="dtw")
    uplot.plot_cluster_centers(kmeans.cluster_centers_)
    return detect_anomalie_2(data, distances)

def detect_feature_anomalie(data, x_, kmeans):
    # Get the distances of each data point to the cluster centers
    distances = kmeans.transform(x_)

    return detect_anomalie_2(data, distances)