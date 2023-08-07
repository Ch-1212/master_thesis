import pandas as pd
import numpy as np
import Saxpy
import Sax_cluster
from sklearn import metrics

df = pd.read_csv('data/data_training.csv')
data = np.array(df['3'])

own_label = pd.read_csv(r'data/own_labels_training.csv')
own_label = own_label['own_labels'][:len(data)]

def test_for_sax_parameters():
    label = pd.DataFrame(index=range(25), columns=range(11))
    ari = pd.DataFrame(index=range(25), columns=range(11))
    ami = pd.DataFrame(index=range(25), columns=range(11))
    fow_mal = pd.DataFrame(index=range(25), columns=range(11))

    for paa in range(2, 13):  # 25
        for alph in range(2, 5):  # 11
            Saxpy.saxpy(data, paa, alph)
            label[alph][paa] = Sax_cluster.cluster_sax(paa, alph)
            ari[alph][paa] = metrics.adjusted_rand_score(own_label, label[alph][paa])
            ami[alph][paa] = metrics.adjusted_mutual_info_score(own_label, label[alph][paa])
            fow_mal[alph][paa] = metrics.fowlkes_mallows_score(own_label, label[alph][paa])

    print(label)

    # Drop rows with all NaN values
    ari = ari.dropna(how='all')
    # Drop columns with all NaN values
    ari = ari.dropna(axis=1, how='all')

    # Drop rows with all NaN values
    ami = ami.dropna(how='all')
    # Drop columns with all NaN values
    ami = ami.dropna(axis=1, how='all')

    # Drop rows with all NaN values
    fow_mal = fow_mal.dropna(how='all')
    # Drop columns with all NaN values
    fow_mal = fow_mal.dropna(axis=1, how='all')

    ari.to_csv(r'data/parameter_results/sax_ari_results.csv')
    ami.to_csv(r'data/parameter_results/sax_ami_results.csv')
    fow_mal.to_csv(r'data/parameter_results/sax_fow_mal_results.csv')
    # ari_max = ari.max()
    # s = ari.max()[ari.max() == ari.max(1).max()].index
    # s = str(s[0])
    # max_index = ari.idxmax()[s]

    # Dann noch Clustering und dann noch ARI, diesen dann vergleichen, am besten in Tabelle
    # Sichergehen, ob erst z-norm gut ist -> Ist gut

def test_for_sax_cluster():
    # Sax clustering parameter test (replace alphabet without meaning vs depending on cut)
    ari = []
    ami = []
    fow_mal = []
    data_labels = Sax_cluster.cluster_sax(12, 3)
    ari.append(metrics.adjusted_rand_score(own_label, data_labels['sax_labels']))
    ami.append(metrics.adjusted_mutual_info_score(own_label, data_labels['sax_labels']))
    fow_mal.append(metrics.fowlkes_mallows_score(own_label, data_labels['sax_labels']))

    ari = [0.702467342189628, 0.526804248412529]
    ami = [0.7448615483793653, 0.6868390964629757]
    fow_mal = [0.8426753211578423, 0.7318062144049411]
    # For all metrics replacing the alphabet on cut yields better results

    sax_cluster_parameter_results = pd.DataFrame([ari, ami, fow_mal]).transpose()
    sax_cluster_parameter_results.columns = ['ARI', 'AMI', 'Fowlkes-Mallows']
    sax_cluster_parameter_results.to_csv(r'data/parameter_results/sax_cluster_parameter_results.csv')

#test_for_sax_parameters()
#test_for_sax_cluster()
print("Finished parameter test")