import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import itertools as it
from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm
import joblib
from sklearn.metrics import silhouette_score
import Anomalie_detection as ad

df = pd.read_csv('data/data_training.csv')
data = np.array(df['3'])
data = data.transpose()

# step*length means that the windows are not overlapping
def moving_window(x, length, step=1):
    streams = it.tee(x, length)
    return zip(*[it.islice(stream, i, None, (step * length)) for stream, i in zip(streams, it.count(step=step))])


# 96 is 24h, 48 is 12h
window_length = 96
x_train = list(moving_window(data, window_length))
x_train = np.asarray(x_train)


def kmeans_cluster(x_train, n_cluster, init, metric, averaging):

    k_means = TimeSeriesKMeans(n_clusters=n_cluster, init_algorithm=rf"{init}", metric=rf"{metric}", averaging_method=rf"{averaging}")
    k_means.fit(x_train)

    # Save fitted model to use later
    #joblib.dump(k_means, "data/fitted_models/kmeans.pkl")

    # Adapt format, so that the plot works
    #plot_cluster_algorithm(k_means, x_train.reshape(len(x_train), 1, len(x_train.transpose())), k_means.n_clusters)

    k_means_label = pd.DataFrame([np.zeros(len(k_means.labels_)), k_means.labels_], dtype=int)
    k_means_label = k_means_label.transpose().rename(columns={0: 'index_real_data', 1: 'labels'})
    for i in range(0, len(k_means_label)):
        k_means_label['index_real_data'][i] = i * 96
    k_means_label.to_csv(rf'data/results_daily_kmeans_training.csv')
    print('Finished k_means sktime')


    list_label = []
    # Add daily label to each data point
    for i in range(0, len(k_means.labels_)):
        label = k_means.labels_[i]
        for j in range(0, window_length):
            list_label.append(label)

    data_labels = np.stack((data[0:len(list_label)], list_label), axis=1)
    data_labels = pd.DataFrame(data_labels, index=df['Timestamp'][0:len(list_label)], columns=['data', 'dtw_labels'])
    #data_labels.to_csv('data/Friedrichshagen/result_kmeans_training.csv')

    # Detect anomalies
    #anomalies_indexes_1_weekly = ad.detect_anomalie_1(data_labels, k_means.labels_)
    #anomalies_indexes_2 = ad.detect_ts_anomalie(data_labels, x_train.reshape(len(x_train), 1, len(x_train.transpose())), k_means)

    # anomalies_indexes_1_weekly = pd.DataFrame(anomalies_indexes_1_weekly)
    # anomalies_indexes_1_weekly.to_csv('data/anomalie_1_kmeans.csv')
    # anomalies_indexes_2 = pd.DataFrame(anomalies_indexes_2)
    # anomalies_indexes_2.to_csv('data/anomalie_2_kmeans.csv')

    return k_means.labels_

kmeans_cluster(x_train, 3, "kmeans++", "dtw", "mean")

print("Finished kmeans")

def elbow_silhouette(x_):
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    elbow = []
    ss = []
    for n_clusters in range_n_clusters:
        # iterating through cluster sizes
        clusterer = TimeSeriesKMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(x_)
        # Finding the average silhouette score
        silhouette_avg = silhouette_score(x_, cluster_labels)
        ss.append(silhouette_avg)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
        # Finding the average SSE"
        elbow.append(clusterer.inertia_)  # Inertia: Sum of distances of samples to their closest cluster center
    fig = plt.figure(figsize=(14, 7))
    plt.rcParams.update({'font.size': 16})
    fig.add_subplot(121)
    plt.plot(range_n_clusters, elbow, 'b-', label='Sum of squared error')
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.legend()
    fig.add_subplot(122)
    plt.plot(range_n_clusters, ss, 'b-', label='Silhouette Score')
    plt.xlabel("Number of cluster")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.show()

#elbow_silhouette(x_train)