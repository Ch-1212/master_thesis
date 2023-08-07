# -*- coding: utf-8 -*-
"""
Created on 2023

@author: charlottebarth
"""

import pandas as pd
import numpy as np
import itertools
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif


# load data to dict derived class Bunch
iris = datasets.load_iris()
target = datasets.load_iris().target


# convert to dataframe for processing
iris = pd.DataFrame(iris.data, columns = iris.feature_names)

def z_score_standardization(series):
    return (series - series.mean()) / series.std()

#Standardize feature data
df_feature_z = z_score_standardization(iris)

def feature_selection():
    #Variance Threshold funktioniert für normierte Daten nicht; aber bei unnormierten
    #Correlation matrix with class labels
    df_feature_z_label = pd.DataFrame(df_feature_z.values, columns=df_feature_z.columns)
    df_feature_z_label['label'] = pd.DataFrame(target)
    corr = df_feature_z_label.corr()

    bestfeatures = SelectKBest(score_func=f_classif, k=1)
    feature_trim = bestfeatures.fit_transform(df_feature_z, target)
    print(bestfeatures.scores_)
    print(bestfeatures.pvalues_)
    print(feature_trim.shape)


def kmeans_feature(feature):
    # ! Muss ja gar kein Time series k means
    X_train_without_z = iris[feature]
    # Standardize features
    X_train = df_feature_z[feature]
    kmeans = KMeans(n_clusters=3).fit(X_train)
    # result = kmeans.predict(X_test)

    labels = np.unique(kmeans.labels_)

    print('Finished k_means sklearn')

    return kmeans

def elbow_silhouette():
    #range_features = [0, 1, 2, 3]
    range_features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    elbow = []
    ss = []
    ari = []
    feature_combinations = []
    for r in range(1, len(range_features) + 1):
        combinations = itertools.combinations(range_features, r)

        # Loop through each combination
        for features in combinations:
            # Run your algorithm with the current combination of features
            kmeans = kmeans_feature(list(features))

            # Store the results in a suitable data structure
            feature_combinations.append(list(features))

            # Finding the average silhouette score
            silhouette_avg = silhouette_score(df_feature_z[list(features)], kmeans.labels_)
            ss.append(silhouette_avg)
            print("For combination ", features, "The average silhouette_score is :", silhouette_avg)
            # Finding the average SSE
            elbow.append(kmeans.inertia_) # Inertia: Sum of distances of samples to their closest cluster center
            # Calculating the Adjusted Rand Index
            ari.append(adjusted_rand_score(kmeans.labels_, target))

    result = pd.DataFrame([feature_combinations, elbow, ss, ari]).transpose()
    result.columns =['Combinations', 'Inertia', 'Silhouette Score', 'Adjusted Rand Index']
    result.to_csv(r'data/Friedrichshagen/feature_parameter_results_iris.csv')
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
    plt.plot(result['Combinations'], result['Silhouette Score'], 'b-', label='Silhouette Score')
    plt.xlabel("Combinations")
    plt.xticks(rotation=45, fontsize=12)
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.show()
    fig = plt.figure(figsize=(14, 7))
    plt.plot(result['Combinations'], result['Adjusted Rand Index'], 'b-', label='Adjusted Rand Index')
    plt.xlabel("Combinations")
    plt.xticks(rotation=45, fontsize=12)
    plt.ylabel("ARI")
    plt.legend()
    plt.show()

#feature_selection()
#elbow_silhouette()
result = pd.read_csv(r'data/parameter_results/feature_parameter_results_iris.csv')
max = result.loc[result['Silhouette Score'] == result['Silhouette Score'].max(), 'Combinations']
#plot_elbow_silhouette(result)
print("Finished")


