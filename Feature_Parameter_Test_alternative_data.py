import pandas as pd
import numpy as np
import itertools
import matplotlib.pylab as plt
import Feature_kmeans_alternative_data as FH_kmeans_features
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sktime.datasets import load_arrow_head
from sklearn.model_selection import train_test_split

X, y = load_arrow_head()
# X: sktime data container, following mtype specification `return_type`
# The time series data for the problem, with n instances
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

def z_score_standardization(series):
    return (series - series.mean()) / series.std()

df_feature = pd.DataFrame()
#slopes = X_train.apply(lambda x: np.polyfit(X_train.index, x, 1)[0])
for i in range(len(X_train)):
    df_feature[i] = X_train['dim_0'][i].describe()
    plt.plot(X['dim_0'][i])
plt.show()
df_feature = df_feature.transpose()
#df_feature['slope'] = pd.DataFrame(slopes)
df_feature = df_feature.drop(['count'], axis=1)

# Standardize features
df_feature_z = z_score_standardization(df_feature)

def elbow_silhouette():
    range_n_features = [0, 1, 2, 3, 4, 5, 6, 7]
    range_features = ['mean', 'std', 'min', '25%', '50%', '75%', 'max'] #, 'slope']
    elbow = []
    ss = []
    ari = []
    feature_combinations = []
    for r in range(1, len(range_features) + 1):
        combinations = itertools.combinations(range_features, r)

        # Loop through each combination
        for features in combinations:
            # Run your algorithm with the current combination of features
            kmeans, data_labels = FH_kmeans_features.kmeans_feature(list(features))

            # Store the results in a suitable data structure
            feature_combinations.append(list(features))

            # Finding the average silhouette score
            silhouette_avg = silhouette_score(df_feature_z[list(features)], kmeans.labels_)
            ss.append(silhouette_avg)
            print("For combination ", features, "The average silhouette_score is :", silhouette_avg)
            # Finding the average SSE
            elbow.append(kmeans.inertia_) # Inertia: Sum of distances of samples to their closest cluster center
            # Calculating the Adjusted Rand Index
            ari.append(adjusted_rand_score(y_train, kmeans.labels_))

    result = pd.DataFrame([feature_combinations, elbow, ss, ari]).transpose()
    result.columns =['Combinations', 'Inertia', 'Silhouette Score', 'Adjusted Rand Index']
    result.to_csv(r'data/Friedrichshagen/feature_parameter_results_other.csv')
    x = np.arange(0, len(feature_combinations) + 1)

def plot_elbow_silhouette(result):
    x = np.arange(0, len(result))
    fig = plt.figure(figsize=(14, 7))
    plt.rcParams.update({'font.size': 16})
    fig.add_subplot(121)
    plt.plot(x, result['Inertia'], 'b-', label='Sum of squared error')
    plt.xlabel("Combinations")
    plt.ylabel("SSE")
    plt.legend()
    fig.add_subplot(122)
    plt.plot(x, result['Silhouette Score'], 'b-', label='Silhouette Score')
    plt.xlabel("Combinations")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.show()
    fig = plt.figure(figsize=(14, 7))
    plt.plot(x, result['Adjusted Rand Index'], 'b-', label='Adjusted Rand Index')
    plt.xlabel("Combinations")
    plt.ylabel("ARI")
    plt.legend()
    plt.show()

#elbow_silhouette()
result = pd.read_csv(r'data/parameter_results/feature_parameter_results_other.csv')
plot_elbow_silhouette(result)


# for i in range_n_features:
#     # iterating through features
#     features = []
#     features.append(range_features[i])
#     for j in range(i + 1, len(range_n_features)):
#         features.append(range_features[j])