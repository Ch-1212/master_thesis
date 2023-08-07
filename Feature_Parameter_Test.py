import pandas as pd
import numpy as np
import itertools
import matplotlib.pylab as plt
import Feature_kmeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score

df_feature_z = pd.read_csv(r'data/data_features.csv')
df_feature_z = df_feature_z.drop(['Unnamed: 0'], axis=1)

own_label = pd.read_csv(r'data/own_labels_training.csv')
own_label = own_label['own_labels']

own_label_daily = pd.read_csv(r'data/own_labels_daily.csv')
own_daily_label = own_label_daily['own_labels']
def feature_selection():
    # Variance Threshold funktioniert nur für unnormierte Daten
    # Correlation matrix with class labels
    df_feature_z_label = pd.DataFrame(df_feature_z.values, columns=df_feature_z.columns)
    df_feature_z_label['label'] = pd.DataFrame(own_daily_label[:len(df_feature_z_label)])
    corr = df_feature_z_label.corr()

    corr = corr.round(4)
    corr.to_csv('data/parameter_results/feature_corr_rounded_parameter_results.csv')

    bestfeatures = SelectKBest(score_func=f_classif, k=1)
    feature_trim = bestfeatures.fit_transform(df_feature_z, own_daily_label[:len(df_feature_z)])
    print(bestfeatures.scores_)
    print(bestfeatures.pvalues_)
    print(feature_trim.shape)
def elbow_silhouette():
    range_n_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    range_features = ['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'slope', 'kurt', 'skew']
    elbow = []
    ss = []
    ari = []
    ami = []
    feature_combinations = []
    for r in range(1, len(range_features) + 1):
        combinations = itertools.combinations(range_features, r)

        # Loop through each combination
        for features in combinations:
            # Run your algorithm with the current combination of features
            kmeans, data_labels = Feature_kmeans.kmeans_feature(list(features))

            # Store the results in a suitable data structure
            feature_combinations.append(list(features))

            # Finding the average silhouette score
            silhouette_avg = silhouette_score(df_feature_z[list(features)], kmeans.labels_)
            ss.append(silhouette_avg)
            print("For combination ", features, "The average silhouette_score is :", silhouette_avg)
            # Finding the average SSE
            elbow.append(kmeans.inertia_) # Inertia: Sum of distances of samples to their closest cluster center
            # Calculating the Adjusted Rand Index
            ari.append(adjusted_rand_score(own_label[:len(data_labels)], data_labels.feature_labels))
            ami.append(adjusted_mutual_info_score(own_label[:len(data_labels)], data_labels.feature_labels))

    result = pd.DataFrame([feature_combinations, elbow, ss, ari, ami]).transpose()
    result.columns =['Combinations', 'Inertia', 'Silhouette Score', 'Adjusted Rand Index', 'Adjusted Mutual Information']
    result.to_csv(r'data/parameter_results/feature_parameter_results.csv')
    x = np.arange(0, len(feature_combinations) + 1)

def plot_elbow_silhouette(result):
    x = np.arange(0, len(result))
    fig = plt.figure(figsize=(14, 7))
    plt.rcParams.update({'font.size': 16})
    fig.add_subplot(121)
    plt.plot(x, result['Inertia'], 'b-', label='Sum of squared error')
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.legend()
    fig.add_subplot(122)
    plt.plot(x, result['Silhouette Score'], 'b-', label='Silhouette Score')
    plt.xlabel("Number of cluster")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.show()
    fig = plt.figure(figsize=(14, 7))
    plt.plot(x, result['Adjusted Rand Index'], 'b-', label='Adjusted Rand Index')
    plt.xlabel("Combinations")
    plt.ylabel("ARI")
    plt.legend()
    plt.show()
    fig = plt.figure(figsize=(14, 7))
    cond = result['Adjusted Rand Index'] >= 0.8
    plt.plot(result['Combinations'][cond], result['Adjusted Rand Index'][cond], 'b-', label='Adjusted Rand Index')
    plt.xlabel("Combinations")
    plt.xticks(rotation=45, fontsize=12)
    plt.ylabel("Adjusted Rand Index")
    plt.legend()
    plt.show()
    fig = plt.figure(figsize=(14, 7))
    plt.plot(x, result['Adjusted Mutual Information'], 'b-', label='Adjusted Mutual Information')
    plt.xlabel("Combinations")
    plt.ylabel("AMI")
    plt.legend()
    plt.show()

#elbow_silhouette()
feature_selection()
result = pd.read_csv(r'data/parameter_results/feature_parameter_results.csv')
result = result.drop(['Unnamed: 0'], axis=1)
result_rounded = result.round(4)
result_rounded.to_csv(r'data/parameter_results/feature_rounded_parameter_results.csv', index=False)
plot_elbow_silhouette(result) # ari, ami and silhouette and selectKbest and corr_with_label propose the 'min' as best result, weirdly the combination of min with something else doesn't seem to improve the results


# for i in range_n_features:
#     # iterating through features
#     features = []
#     features.append(range_features[i])
#     for j in range(i + 1, len(range_n_features)):
#         features.append(range_features[j])