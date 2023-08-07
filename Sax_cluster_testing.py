import pandas as pd
import numpy as np
import joblib
import Anomalie_detection as ad

def cluster_sax(paa, alphabet_size):
    results_saxpy = pd.read_csv(rf'data/saxpy_results/paa_{paa}_alphabet_{alphabet_size}_testing.csv')

    # Split SAX word into different columns
    data_symbols = results_saxpy['symbols'].apply(str)
    df_symbols = pd.DataFrame(data_symbols.apply(lambda x: list(x)).tolist())

    # Replace symbols with numerical values
    def replace_alphabet_depending_on_cut():
        char_to_float = {'a': -1.5, 'b': -0.3372449, 'c': 0.3372449, 'd': 1.5}
        # Replace every character with a specific float value depending on the character
        df = df_symbols.replace(list(char_to_float.keys()), list(char_to_float.values()), regex=True)
        df = np.asarray(df)
        return df

    x_test = replace_alphabet_depending_on_cut()

    # Load fitted model (on training data)
    k_means = joblib.load("data/fitted_models/sax_kmeans.pkl")

    k_predict = k_means.predict(x_test)

    # cluster centers (back to symbols): 1: cccccbbba; 2: abcccccbb; 3: aabbcccdd; 4: abbccccc; 5: ccbbbbbcc

    print('Finished cluster saxpy')


    # Get results back to original data
    # Import original data and sax_none
    df = pd.read_csv('data/data_testing.csv')
    data = np.array(df['3'])
    sax_none = pd.read_csv(rf'data/saxpy_results/paa_{paa}_alphabet_{alphabet_size}_sax_none_testing.csv')
    paa_coef = pd.read_csv('data/saxpy_results/paa_coef_testing.csv')
    paa_coef = np.asarray(paa_coef.drop(['Unnamed: 0'], axis=1))

    list_label = [5] * len(data)
    # Add label to data
    for i in range(0, len(k_predict)):
        key = data_symbols[i]
        label = k_predict[i]
        indices = sax_none[sax_none['Unnamed: 0'] == key]['0']
        indices = indices.reset_index()['0'].str.replace('[\[\]]', '')
        indices = indices.str.split(',', expand=True)
        # Change data type of all values in DataFrame to integer
        indices = indices.astype(int)
        # Make sure that found indices fit to the indices in the data
        indices = indices - 22176
        for j in range(0, len(indices.transpose())):
            for l in range(96):
                list_label[int(indices[j][0]) + l] = label

    # The data points that don't belong to any sliding window (last half day)
    if 5 in list_label:
        print('Some indices are missing')

    pd.DataFrame(list_label).to_csv(rf'data/saxpy_results/label_sax_kmeans_paa_{paa}_alph_{alphabet_size}_testing.csv')

    data_labels = np.stack((data[0:len(list_label)], list_label), axis=1)
    data_labels = pd.DataFrame(data_labels, index=df['Timestamp'][0:len(list_label)], columns=['data', 'sax_labels'])
    data_labels.to_csv(rf'data/results/result_sax_kmeans_paa_{paa}_alph_{alphabet_size}_testing.csv')

    # Detect anomalies
    # Get daily labels for anomalie_1 detection
    def label_to_daily_label():
        daily_label = []
        for i in range(0, len(list_label)-96, 96):
            daily_label.append(list_label[i])
        return daily_label

    anomalies_indexes_1_weekly = ad.detect_anomalie_1(data_labels, label_to_daily_label())
    # Use paa_coef for distance calculation for anomalie_2 detection
    anomalies_indexes_2 = ad.detect_ts_anomalie(data_labels, paa_coef.reshape(len(paa_coef), 1, len(paa_coef.transpose())), k_means)

    # anomalies_indexes_1_weekly = pd.DataFrame(anomalies_indexes_1_weekly)
    # anomalies_indexes_1_weekly.to_csv('data/results/anomalie_1_sax_kmeans.csv')
    # anomalies_indexes_2 = pd.DataFrame(anomalies_indexes_2)
    # anomalies_indexes_2.to_csv('data/results/anomalie_2_sax_kmeans.csv')

    return list_label

cluster_sax(paa=12, alphabet_size=3)

print("Sax_cluster_testing finished")