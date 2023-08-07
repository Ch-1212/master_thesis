import pandas as pd
import numpy as np
import itertools as it
from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm
import joblib
import Anomalie_detection as ad

df = pd.read_csv('data/data_testing.csv')

# step*length means that the windows are not overlapping
def moving_window(x, length, step=1):
    streams = it.tee(x, length)
    return zip(*[it.islice(stream, i, None, (step * length)) for stream, i in zip(streams, it.count(step=step))])


# 96 is 24h, 48 is 12h
window_length = 96
x_test = list(moving_window(df['3'].transpose(), window_length))
x_test = np.asarray(x_test)

def kmeans_cluster():

    # Load fitted model (on training data)
    k_means = joblib.load("data/fitted_models/kmeans.pkl")

    k_predict = k_means.predict(x_test)

    # Adapt format, so that the plot works
    #plot_cluster_algorithm(k_means, x_test.reshape(len(x_test), 1, len(x_test.transpose())), k_means.n_clusters)

    print('Finished k_means sktime')

    list_label = []
    # Add daily label to each data point
    for i in range(0, len(k_predict)):
        label = k_predict[i]
        for j in range(0, window_length):
            list_label.append(label)

    data_labels = np.stack((df['3'][0:len(list_label)], list_label), axis=1)
    data_labels = pd.DataFrame(data_labels, index=df['Timestamp'][0:len(list_label)], columns=['data', 'dtw_labels'])
    #data_labels.to_csv(rf'data/results/result_kmeans_testing.csv')

    # Detect anomalies
    anomalies_indexes_1_weekly = ad.detect_anomalie_1(data_labels, k_predict)
    anomalies_indexes_2 = ad.detect_ts_anomalie(data_labels, x_test.reshape(len(x_test), 1, len(x_test.transpose())), k_means)

    # anomalies_indexes_1_weekly = pd.DataFrame(anomalies_indexes_1_weekly)
    # anomalies_indexes_1_weekly.to_csv(rf'data/results/anomalie_1_kmeans.csv')
    # anomalies_indexes_2 = pd.DataFrame(anomalies_indexes_2)
    # anomalies_indexes_2.to_csv(rf'data/results/anomalie_2_kmeans.csv')

    return k_means.labels_

kmeans_cluster()

print("Finished kmeans_testing")