import pandas as pd
import numpy as np
import itertools as it
import Kmeans
from sklearn import metrics

window_parameter_result = pd.read_csv(r'data/Friedrichshagen/window_parameter_results.csv')

def window_parameter():
    df = pd.read_csv('data/data_training.csv')
    data = np.array(df['3'])
    data = data.transpose()

    own_label = pd.read_csv(r'data/own_labels_training.csv')

    ari = []
    ami = []
    fow_mal = []

    # 96 is 24h
    window_length = 96

    # step*length means that the windows are not overlapping
    def moving_window(x, length, step=1):
        streams = it.tee(x, length)
        return zip(*[it.islice(stream, i, None, step * length) for stream, i in zip(streams, it.count(step=step))])

    # Keine Überlappung
    x_train = list(moving_window(data, window_length))
    x_train = np.asarray(x_train)

    labels_ = Kmeans.kmeans_cluster(x_train, 3, "kmeans++", "dtw", "mean")

    # for non-overlapping windows
    label = []
    # Add label to data
    for i in range(0, len(labels_)):
        labels = labels_[i]
        for j in range(0, 96):  # window_length):
            label.append(labels)

    ari.append(metrics.adjusted_rand_score(own_label.own_labels[:len(label)], label))
    ami.append(metrics.adjusted_mutual_info_score(own_label.own_labels[:len(label)], label))
    fow_mal.append(metrics.fowlkes_mallows_score(own_label.own_labels[:len(label)], label))

    def moving_window(x, length, step=1):
        streams = it.tee(x, length)
        return zip(*[it.islice(stream, i, None, step * 72) for stream, i in zip(streams, it.count(step=step))])

    # 25% Überlappung
    x_train = list(moving_window(data, window_length))
    x_train = np.asarray(x_train)

    labels_ = Kmeans.kmeans_cluster(x_train, 3, "kmeans++", "dtw", "mean")

    label = []
    # Add label to data
    for i in range(0, len(labels_)):
        labels = labels_[i]
        for j in range(0, 72):  # window_length):
            label.append(labels)

    ari.append(metrics.adjusted_rand_score(own_label.own_labels[:len(label)], label))
    ami.append(metrics.adjusted_mutual_info_score(own_label.own_labels[:len(label)], label))
    fow_mal.append(metrics.fowlkes_mallows_score(own_label.own_labels[:len(label)], label))

    # step*length means that the windows are not overlapping
    def moving_window(x, length, step=1):
        streams = it.tee(x, length)
        return zip(*[it.islice(stream, i, None, step * 48) for stream, i in zip(streams, it.count(step=step))])

    # 50% Überlappung
    x_train = list(moving_window(data, window_length))
    x_train = np.asarray(x_train)

    labels_ = Kmeans.kmeans_cluster(x_train, 3, "kmeans++", "dtw", "mean")

    label = []
    # Add label to data
    for i in range(0, len(labels_)):
        labels = labels_[i]
        for j in range(0, 48):  # window_length):
            label.append(labels)

    ari.append(metrics.adjusted_rand_score(own_label.own_labels[:len(label)], label))
    ami.append(metrics.adjusted_mutual_info_score(own_label.own_labels[:len(label)], label))
    fow_mal.append(metrics.fowlkes_mallows_score(own_label.own_labels[:len(label)], label))

    def moving_window(x, length, step=1):
        streams = it.tee(x, length)
        return zip(*[it.islice(stream, i, None, step * 24) for stream, i in zip(streams, it.count(step=step))])

    # 75% Überlappung
    x_train = list(moving_window(data, window_length))
    x_train = np.asarray(x_train)

    labels_ = Kmeans.kmeans_cluster(x_train, 3, "kmeans++", "dtw", "mean")

    label = []
    # Add label to data
    for i in range(0, len(labels_)):
        labels = labels_[i]
        for j in range(0, 24):  # window_length):
            label.append(labels)

    ari.append(metrics.adjusted_rand_score(own_label.own_labels[:len(label)], label))
    ami.append(metrics.adjusted_mutual_info_score(own_label.own_labels[:len(label)], label))
    fow_mal.append(metrics.fowlkes_mallows_score(own_label.own_labels[:len(label)], label))

    def moving_window(x, length, step=1):
        streams = it.tee(x, length)
        return zip(*[it.islice(stream, i, None, step) for stream, i in zip(streams, it.count(step=step))])

    # Volle Überlappung
    x_train = list(moving_window(data, window_length, 1))
    x_train = np.asarray(x_train)

    labels_ = Kmeans.kmeans_cluster(x_train, 3, "kmeans++", "dtw", "mean")

    label = []
    # Add label to data
    for i in range(0, len(labels_)):
        labels = labels_[i]
        for j in range(0, 1):  # window_length):
            label.append(labels)

    ari.append(metrics.adjusted_rand_score(own_label.own_labels[:len(label)], label))
    ami.append(metrics.adjusted_mutual_info_score(own_label.own_labels[:len(label)], label))
    fow_mal.append(metrics.fowlkes_mallows_score(own_label.own_labels[:len(label)], label))

    return ari, ami, fow_mal

ari, ami, fow_mal = window_parameter()
print('Finished Window Parameter Test')

#ami = [0.9414662493790259, 0.40970796803317633, 0.37117397328035434, 0.34720697967700315, 0.29018258567351873]
#ari = [0.9646961406941534, 0.3898573556104788, 0.31250702453939877, 0.35302404438776736, 0.2425061903048079]
#fow_mal = [0.982283671778268, 0.650504412612862, 0.6020649910707688, 0.6285896546585267, 0.5607086424272462]

window_parameter_results = pd.DataFrame([ari, ami, fow_mal]).transpose()
window_parameter_results.columns = ['ARI', 'AMI', 'Fowlkes-Mallows']
window_parameter_results.to_csv(r'data/parameter_results/window_parameter_results.csv')