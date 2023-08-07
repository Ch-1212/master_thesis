# -*- coding: utf-8 -*-
"""
Created on 2023

@author: charlottebarth
"""


import scipy.io


BOHR = scipy.io.loadmat('data/BOHRMAS.mat')
ORIG = BOHR['ORIG']

# Für die Absicherung der Serienproduktion von Bohrmaschinenmotoren wird eine
# Stichprobe gezogen, um einen Klassifikator zu entwerfen.
# Die Stichprobe umfasst  insgesamt 219 Bohrmaschinenmotoren 
# mit folgenden 5 Klassen
# GUT N1=40
# LAR Lärm Ankerritzel N2=30
# RUF:Rundfeuer N3=50 
# SSS:Schlagstelle Stirnrad N4=49
# UWA:Unwucht Anker N5=50
class_names = ('GUT', 'LAR', 'RUF', 'SSS', 'UWA')
K = [40, 30, 50, 49, 50]

# Insgesamt 18 Merkmale wurden wie folgt aus dem Motorstrom erzeugt:
# 1.-12. Merkmal: Teilbandleistungen
# 13. Merkmal: Effektivwert
# 14. Merkmal: Schiefe
# 15. Merkmal: Wölbung
# 16. Merkmal: Streuung der Einhüllenden
# 17. Merkmal: Schiefe der Einhüllenden
# 18. Merkmal: Wölbung der Einhüllenden

# ORIG ist eine Matrix mit 18 Zeilen (Merkmale) und 219 Spalten (Motoren)

import pandas as pd
import numpy as np
import itertools
import matplotlib.pylab as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sktime.datasets import load_arrow_head
from sklearn.model_selection import train_test_split

def z_score_standardization(series):
    return (series - series.mean()) / series.std()

cluster = np.repeat([1, 2, 3, 4, 5], K)
#cluster = pd.DataFrame({'value': data})
#Standardize feature data
ORIG = pd.DataFrame(ORIG)
ORIG_z = z_score_standardization(ORIG.transpose())
ORIG_z = pd.DataFrame(ORIG_z)

def feature_selection():
    #Variance Threshold funktioniert für normierte Daten nicht; aber bei unnormierten
    #Correlation matrix with class labels
    df_feature_z_label = pd.DataFrame(ORIG_z.values, columns=ORIG_z.columns)
    df_feature_z_label['label'] = pd.DataFrame(cluster)
    corr = df_feature_z_label.corr()

    bestfeatures = SelectKBest(score_func=f_classif, k=1)
    feature_trim = bestfeatures.fit_transform(ORIG_z, cluster)
    print(bestfeatures.scores_)
    print(bestfeatures.pvalues_)
    print(feature_trim.shape)

def kmeans_feature(feature):
    # ! Muss ja gar kein Time series k means
    X_train_without_z = ORIG.transpose()[feature]
    # Standardize features
    X_train = ORIG_z[feature]
    #X_train = ORIG[feature].transpose()
    kmeans = KMeans(n_clusters=5).fit(X_train)
    # result = kmeans.predict(X_test)

    # anomalies_indexes_1 = ad.detect_anomalie_1(kmeans.labels_)
    # anomalies_indexes_2 = ad.detect_feature_anomalie(X_train_without_z, X_train, kmeans)

    labels = np.unique(kmeans.labels_)

    print('Finished k_means sklearn')

    return kmeans

def elbow_silhouette():
    range_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] #0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    #range_features = ['mean', 'std', 'min', '25%', '50%', '75%', 'max', ] #, 'slope']
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
            silhouette_avg = silhouette_score(ORIG_z[list(features)], kmeans.labels_)
            ss.append(silhouette_avg)
            print("For combination ", features, "The average silhouette_score is :", silhouette_avg)
            # Finding the average SSE
            elbow.append(kmeans.inertia_) # Inertia: Sum of distances of samples to their closest cluster center
            # Calculating the Adjusted Rand Index
            ari.append(adjusted_rand_score(kmeans.labels_, cluster))

    result = pd.DataFrame([feature_combinations, elbow, ss, ari]).transpose()
    result.columns =['Combinations', 'Inertia', 'Silhouette Score', 'Adjusted Rand Index']
    result.to_csv(r'data/Friedrichshagen/feature_parameter_results_daten.csv')
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
    cond = result['Silhouette Score'] >= 0.6
    plt.plot(result['Combinations'][cond], result['Silhouette Score'][cond], 'b-', label='Silhouette Score')
    plt.xlabel("Combinations")
    plt.xticks(rotation=45, fontsize=12)
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.show()
    fig = plt.figure(figsize=(14, 7))
    plt.plot(x, result['Adjusted Rand Index'], 'b-', label='Adjusted Rand Index')
    plt.xlabel("Combinations")
    #plt.xticks(rotation=45, fontsize=12)
    plt.ylabel("ARI")
    plt.legend()
    plt.show()
    fig = plt.figure(figsize=(14, 7))
    cond = result['Adjusted Rand Index'] >= 0.65
    plt.plot(result['Combinations'][cond], result['Adjusted Rand Index'][cond], 'b-', label='Adjusted Rand Index')
    plt.xlabel("Combinations")
    plt.xticks(rotation=45, fontsize=12)
    plt.ylabel("ARI")
    plt.legend()
    plt.show()

#feature_selection()
#elbow_silhouette()
result = pd.read_csv(r'data/parameter_results/feature_parameter_results_daten.csv')
result = result.drop(['Unnamed: 0'], axis=1)
result_round = result.round(4)
result_round.to_csv(r'data/parameter_results/feature_parameter_results_drill_motor_rounded.csv', index=False)
max = result.loc[result['Silhouette Score'] == result['Silhouette Score'].max(), 'Combinations']
#plot_elbow_silhouette(result)
print("Finished")


