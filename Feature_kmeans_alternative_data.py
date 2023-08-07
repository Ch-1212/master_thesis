import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import itertools as it
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import util_plots as uplt
import Anomalie_detection as ad
from sktime.datasets import load_arrow_head
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/data_training.csv')
data = np.array(df['3'])

X, y = load_arrow_head()
# X: sktime data container, following mtype specification `return_type`
# The time series data for the problem, with n instances
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)


#step*length means that the windows are not overlapping
def moving_window(x, length, step=1):
    streams = it.tee(x, length)
    return zip(*[it.islice(stream, i, None, step * length) for stream, i in zip(streams, it.count(step=step))])

def z_score_standardization(series):
    return (series - series.mean()) / series.std()

#all three clusters
plt.figure()
plt.plot(X['dim_0'][0])
plt.plot(X['dim_0'][1])
plt.plot(X['dim_0'][2])
plt.show()

#cluster 2
plt.figure()
plt.plot(X['dim_0'][1])
plt.plot(X['dim_0'][4])
plt.plot(X['dim_0'][7])
plt.show()

plt.figure()
plt.plot(X['dim_0'][2])
plt.plot(X['dim_0'][5])
plt.plot(X['dim_0'][8])
plt.show()


# Slopes
df_feature = pd.DataFrame()
#slopes = X_train.apply(lambda x: np.polyfit(X_train.index, x, 1)[0])
for i in range(len(X_train)):
    df_feature[i] = X_train['dim_0'][i].describe()
    plt.plot(X['dim_0'][i])
    plt.show()
df_feature = df_feature.transpose()
#df_feature['slope'] = pd.DataFrame(slopes)
df_feature = df_feature.drop(['count'], axis=1)

# Variance
df_varianz = df_feature.var()
# Correlation (pearson's coefficient)
corr_matrix = df_feature.corr()

# Standardize features
df_feature_z = z_score_standardization(df_feature)

uplt.plot_features(df_feature_z)

def kmeans_feature(feature):
    # ! Muss ja gar kein Time series k means
    X_train_without_z = df_feature[feature]
    X_train = df_feature_z[feature]
    kmeans = KMeans(n_clusters=3).fit(X_train)
    # result = kmeans.predict(X_test)

    #anomalies_indexes_1 = ad.detect_anomalie_1(kmeans.labels_)
    #anomalies_indexes_2 = ad.detect_feature_anomalie(X_train_without_z, X_train, kmeans)

    # Cluster membership
    _labels = pd.Series(kmeans.labels_, index=df_feature_z.index)

    labels = np.unique(kmeans.labels_)

    print('Finished k_means sklearn')

    list_label = []
    # Add label to data
    for i in range(0, len(kmeans.labels_)):
        label = kmeans.labels_[i]
        for j in range(0, 96):
            list_label.append(label)

    # ! data has more points than list_label (96x327): vermutlich ignoriert sliding window die letzten Daten
    data_labels = np.stack((data[0:len(list_label)], list_label), axis=1)
    data_labels = pd.DataFrame(data_labels, columns=['data', 'feature_labels'])
    # data_labels.to_csv(r'data/Friedrichshagen/result_feature.csv')
    # anomalies_indexes_1 = pd.DataFrame(anomalies_indexes_1)
    # anomalies_indexes_1.to_csv(r'data/Friedrichshagen/anomalie_1_feature.csv')
    # anomalies_indexes_2 = pd.DataFrame(anomalies_indexes_2)
    # anomalies_indexes_2.to_csv(r'data/Friedrichshagen/anomalie_2_feature.csv')

    return kmeans, data_labels


#kmeans_feature(['mean', 'std', 'slope'])
#kmeans_feature(['25%'])