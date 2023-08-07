import pandas as pd
import numpy as np
import itertools as it
import Kmeans
from sklearn import metrics

df = pd.read_csv('data/data_training.csv')
data = np.array(df['3'])
data = data.transpose()

own_labels = pd.read_csv(r'data/own_labels_training.csv')
own_labels = own_labels['own_labels']

list_metric = ["euclidean", "dtw"]
list_init = ["forgy", "random", "kmeans++"]
list_averaging = ["mean", "dba"]

ari = []
ami = []
fow_mal = []

# 96 is 24h
window_length = 96

#step*length means that the windows are not overlapping
def moving_window(x, length, step=1):
    streams = it.tee(x, length)
    return zip(*[it.islice(stream, i, None, step * length) for stream, i in zip(streams, it.count(step=step))])

x_train = list(moving_window(data, window_length))
x_train = np.asarray(x_train)

for i in range(len(list_init)):
#for i in range(len(list_metric)):
#for i in range(len(list_averaging)):
    labels_ = Kmeans.kmeans_cluster(x_train, 3, list_init[i], "dtw", "mean")
    #labels_ = Kmeans.kmeans_cluster(x_train, 3, "kmeans++", list_metric[i], "mean")
    #labels_ = Kmeans.kmeans_cluster(x_train, 3, "kmeans++", "dtw", list_averaging[i])

    label = []
    # Add label to data
    for j in range(0, len(labels_)):
        labels = labels_[j]
        for j in range(0, 96):
            label.append(labels)

    ari.append(metrics.adjusted_rand_score(own_labels[:len(label)], label))
    ami.append(metrics.adjusted_mutual_info_score(own_labels[:len(label)], label))
    fow_mal.append(metrics.fowlkes_mallows_score(own_labels[:len(label)], label))

kmeans_parameter_results = pd.DataFrame([ari, ami, fow_mal]).transpose()
kmeans_parameter_results.columns = ['ARI', 'AMI', 'Fowlkes-Mallows']
kmeans_parameter_results.to_csv(r'data/parameter_results/kmeans_init_parameter_results.csv')