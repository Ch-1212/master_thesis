import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import itertools as it
import util_plots as uplt
import joblib
import Anomalie_detection as ad

df = pd.read_csv('data/data_testing.csv')
data = np.array(df['3'])
data = data.transpose()

# step*length means that the windows are not overlapping
def moving_window(x, length, step=1):
    streams = it.tee(x, length)
    return zip(*[it.islice(stream, i, None, step * length) for stream, i in zip(streams, it.count(step=step))])

# Could have used the Sklearn.preprocessing.StandardScaler for this
def z_score_standardization(series):
    return (series - series.mean()) / series.std()

x_ = list(moving_window(data, 96))
x_ = np.asarray(x_)
x_tr = x_.transpose()

df_sl_wind = pd.DataFrame(np.squeeze(x_tr))

# Slopes
slopes = df_sl_wind.apply(lambda x: np.polyfit(df_sl_wind.index, x, 1)[0])
df_feature = df_sl_wind.describe()
df_feature = df_feature.transpose()
# df_feature['slope'] = pd.DataFrame(slopes)
# df_feature['kurt'] = df_sl_wind.kurtosis()
# df_feature['skew'] = df_sl_wind.skew()
df_feature = df_feature.drop(['count'], axis=1)

# Standardize features
df_feature_z = z_score_standardization(df_feature)
#df_feature_z.to_csv(r'data/data_features_testing.csv')

#uplt.plot_features(df_feature_z)
def plot_features():
    plt.figure()
    plt.rcParams.update({'font.size': 16})
    plt.scatter(df_feature['mean'], df_feature['std'])
    plt.xlabel("mean")
    plt.ylabel("std")
    plt.legend()
    plt.show()

def plot_feature_result(X_train, feature, kmeans, labels):
    # Only works for two features
    plt.figure()
    plt.rcParams.update({'font.size': 16})
    scatter = plt.scatter(X_train[feature[0]], X_train[feature[1]], c=kmeans.labels_, label=labels)
    handles, _ = scatter.legend_elements(num=len(labels))
    plt.xlabel(feature[0])
    plt.ylabel(feature[1])
    plt.legend(handles, labels)
    plt.show()

def kmeans_feature(feature):
    x_test_without_z = df_feature[feature]
    x_test = df_feature_z[feature]

    # Load fitted model (on training data)
    k_means = joblib.load("data/fitted_models/feature_kmeans.pkl")

    k_predict = k_means.predict(x_test)

    # Cluster membership
    _labels = pd.Series(k_predict, index=df_feature_z.index)

    print('Finished k_means sklearn')

    list_label = []
    # Add label to data
    for i in range(0, len(k_predict)):
        label = k_predict[i]
        for j in range(0, 96):
            list_label.append(label)

    data_labels = np.stack((data[0:len(list_label)], list_label), axis=1)
    data_labels = pd.DataFrame(data_labels, index=df['Timestamp'][0:len(list_label)], columns=['data', 'feature_labels'])
    #data_labels.to_csv(r'data/results/result_feature_testing.csv')

    # Detect anomalies
    anomalies_indexes_1 = ad.detect_anomalie_1(data_labels, k_predict)
    anomalies_indexes_2 = ad.detect_feature_anomalie(data_labels, x_test, k_means)

    # anomalies_indexes_1 = pd.DataFrame(anomalies_indexes_1)
    # anomalies_indexes_1.to_csv(r'data/results/anomalie_1_feature.csv')
    # anomalies_indexes_2 = pd.DataFrame(anomalies_indexes_2)
    # anomalies_indexes_2.to_csv(r'data/results/anomalie_2_feature.csv')

    return k_means, data_labels


#kmeans_feature(['mean', 'std', 'slope'])
kmeans_feature(['min'])

print("Finished feature_kmeans_testing")


# def centroid_analysis():
#     clusters = ['Cluster {}'.format(i) for i in range(3)]
#     Centroids_orig = pd.DataFrame(0.0, index=clusters,
#                                   columns=X_train_without_z.columns)
#
#     Centroids_std = pd.DataFrame(0.0, index=clusters,
#                                  columns=X_train.columns)
#     for i in range(3):
#         BM = _labels == i
#         Centroids_orig.iloc[i] = X_train_without_z[BM].median(axis=0)
#         Centroids_std.iloc[i] = X_train[BM].mean(axis=0)
#
#     #displays the average value of each feature for each cluster in a color-coded format. This allows for easy visualization of the differences in feature values across clusters
#     plt.figure(figsize=(10, 3))
#     sns.heatmap(Centroids_std, linewidths=.5, annot=True,
#                 cmap='Purples')
#     plt.show()
#
# centroid_analysis()