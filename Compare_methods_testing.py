import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics import silhouette_score
import util_plots as uplot

# own_daily_label = pd.read_csv(r'data/Friedrichshagen/own_daily_labels.csv')
own_labels = pd.read_csv(r'data/own_labels_testing.csv')
own_labels = own_labels['own_labels']


results_kmeans = pd.read_csv(r'data/results/result_kmeans_testing.csv')
results_feature = pd.read_csv(r'data/results/result_feature_testing.csv')
results_sax_kmeans = pd.read_csv(r'data/results/result_sax_kmeans_paa_12_alph_3_testing.csv')
results_sax_kmeans = results_sax_kmeans[:len(results_kmeans)]
results_kmeans['Timestamp'] = pd.to_datetime(results_kmeans['Timestamp'])
results_feature['Timestamp'] = pd.to_datetime(results_feature['Timestamp'])
results_sax_kmeans['Timestamp'] = pd.to_datetime(results_sax_kmeans['Timestamp'])

anomalie_1_kmeans = pd.read_csv(r'data/results/anomalie_1_kmeans.csv')
anomalie_2_kmeans = pd.read_csv(r'data/results/anomalie_2_kmeans.csv')
anomalie_1_feature = pd.read_csv(r'data/results/anomalie_1_feature.csv')
anomalie_2_feature = pd.read_csv(r'data/results/anomalie_2_feature.csv')
anomalie_1_sax_kmeans = pd.read_csv(r'data/results/anomalie_1_sax_kmeans.csv')
anomalie_2_sax_kmeans = pd.read_csv(r'data/results/anomalie_2_sax_kmeans.csv')

# Plot original data with clustering results
#uplot.plot_clustered_data(np.array(results_kmeans, dtype=object))
uplot.plot_clustered_data(np.array(results_feature, dtype=object))
#uplot.plot_clustered_data(np.array(results_sax_kmeans, dtype=object))
#uplot.plot_clustered_data_with_anomalies(np.array(results_kmeans), anomalie_1_kmeans, anomalie_2_kmeans)
#uplot.plot_clustered_data_with_anomalies(np.array(results_feature), anomalie_1_feature, anomalie_2_feature)
#uplot.plot_clustered_data_with_anomalies(np.array(results_sax_kmeans), anomalie_1_sax_kmeans, anomalie_2_sax_kmeans)

#Calculate Adjusted Rand Score to see overlapping from own_labels to other results
ari_dtw_kmeans = adjusted_rand_score(own_labels[:len(results_kmeans.dtw_labels)], results_kmeans.dtw_labels)
ari_feature_kmeans = adjusted_rand_score(own_labels[:len(results_feature.feature_labels)], results_feature.feature_labels)
ari_sax_kmeans = adjusted_rand_score(own_labels[:len(results_sax_kmeans.sax_labels)], results_sax_kmeans.sax_labels)

ami_kmeans_dtw = adjusted_mutual_info_score(own_labels[:len(results_kmeans.dtw_labels)], results_kmeans.dtw_labels)
ami_kmeans_feature = adjusted_mutual_info_score(own_labels[:len(results_feature.feature_labels)], results_feature.feature_labels)
ami_kmeans_sax = adjusted_mutual_info_score(own_labels[:len(results_sax_kmeans.sax_labels)], results_sax_kmeans.sax_labels)

#v_measure_kmeans = v_measure_score(own_labels[:len(results_kmeans_euc.dtw_labels)], results_kmeans_euc.dtw_labels) #same as ami?
fowlkes_mallows_kmeans = fowlkes_mallows_score(own_labels[:len(results_kmeans.dtw_labels)], results_kmeans.dtw_labels)
fowlkes_mallows_kmeans_feature = fowlkes_mallows_score(own_labels[:len(results_feature.feature_labels)], results_feature.feature_labels)
fowlkes_mallows_kmeans_sax = fowlkes_mallows_score(own_labels[:len(results_sax_kmeans.sax_labels)], results_sax_kmeans.sax_labels)

silhouette_kmeans = silhouette_score(np.array(results_kmeans.data).reshape(-1,1), results_kmeans.dtw_labels)
plt.rcParams.update({'font.size': 16})

#Feature - DTW
contingency_tbl = pd.crosstab(results_feature.feature_labels, results_kmeans.dtw_labels)
probablity_tbl = contingency_tbl/ contingency_tbl.sum()
plt.figure()
sns.heatmap(probablity_tbl, annot=True, center=0.5, cmap="Greys")
plt.show()
# Plot shows that two clusters found are nearly identical, other two completely different

#DTW - SAX
contingency_tbl = pd.crosstab(results_kmeans.dtw_labels, results_sax_kmeans.sax_labels)
probablity_tbl = contingency_tbl/ contingency_tbl.sum()
plt.figure()
sns.heatmap(probablity_tbl, annot=True, center=0.5, cmap="Greys")
plt.show()

#DTW - SAX
contingency_tbl = pd.crosstab(results_feature.feature_labels, results_sax_kmeans.sax_labels)
probablity_tbl = contingency_tbl/ contingency_tbl.sum()
plt.figure()
sns.heatmap(probablity_tbl, annot=True, center=0.5, cmap="Greys")
plt.show()

#SAX - Label
contingency_tbl = pd.crosstab(results_sax_kmeans.sax_labels, own_labels)
probablity_tbl = contingency_tbl/ contingency_tbl.sum()
plt.figure()
sns.heatmap(probablity_tbl, annot=True, center=0.5, cmap="Greys")
plt.show()

print('Finished compare_methods')