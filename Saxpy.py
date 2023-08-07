import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import saxpy.sax as sax_py
from collections import defaultdict
from saxpy.znorm import znorm
from saxpy.paa import paa

df = pd.read_csv('data/data.csv')
data = np.array(df['3'])
window_size = 96

# Edit original function to allow for non-overlapping windows and znormalization of original data + allow to save paa_coef (interesting for anomalie_detection)
def sax_via_window(series, win_size, paa_size, alphabet_size=3,
                   nr_strategy='exact', z_threshold=0.01):
    """Simple via window conversion implementation."""
    cuts = sax_py.cuts_for_asize(alphabet_size)
    sax = defaultdict(list)
    paa_coef = []

    prev_word = ''

    counter = 0

    zn_series = znorm(series, z_threshold)

    for i in range(0, len(series) - win_size, win_size):

        counter += 1

        zn = zn_series[i:(i+win_size)]

        # zn = znorm(sub_section, z_threshold)

        paa_rep = paa(zn, paa_size)

        paa_coef.append(paa_rep)

        curr_word = sax_py.ts_to_string(paa_rep, cuts)

        if '' != prev_word:
            if 'exact' == nr_strategy and prev_word == curr_word:
                continue
            elif 'mindist' == nr_strategy and\
                    sax_py.is_mindist_zero(prev_word, curr_word):
                continue

        prev_word = curr_word

        sax[curr_word].append(i)

    print(counter)

    return sax, paa_coef

def sax_none_exploration(sax_none):
    list_keys = []
    list_key_length = []

    for i in range(0, len(sax_none.keys())):
        list_keys.append(list(sax_none.keys())[i])
        list_key_length.append(len(sax_none[list(sax_none.keys())[i]]))

    key_length_df = pd.DataFrame(list(zip(list_keys, list_key_length)))
    key_length_df.columns = ['symbols', '#']
    key_length_df_sorted = key_length_df.sort_values('#', ascending=False)
    key_length_sum = np.sum(list_key_length)  # 7
    key_length_df_sorted['%'] = key_length_df_sorted['#'] / key_length_sum
    key_length_df_sorted['sum%'] = key_length_df_sorted['%'].cumsum()
    key_length_sorted = key_length_df_sorted.reset_index()

    def plot_sax_words():
        plt.figure()
        for i in range(0, 9):
            key = key_length_sorted['symbols'][i]
            plt.subplot(3, 3, i + 1)
            plt.ylim(-14, 4)
            plt.title(key)
            for j in range(0, len(sax_none[key])):
                plt.plot(data[sax_none[key][j]:sax_none[key][j] + window_size])
            plt.xlabel('Window')
            plt.ylabel('Temperature [Celsius]')
        plt.tight_layout()
        # plt.savefig(rf'data/Friedrichshagen/saxpy_results/paa_{paa}_alphabet_{alphabet_size}.svg') #, format='png')
        plt.show()


    def plot_cluster_amount():
        fig1 = plt.figure()
        plt.xlabel('Key')
        plt.ylabel('Cummulated data [%]')
        plt.plot(key_length_sorted['symbols'], key_length_sorted['sum%'])
        plt.tight_layout()
        # fig.savefig(rf'data/Friedrichshagen/saxpy_results/paa_{paa}_alphabet_{alphabet_size}', format='svg')
        plt.show()

    # plot_sax_words()
    # plot_cluster_amount()

    return key_length_sorted


def saxpy(datas, paa, alphabet_size):
    # Use z-normalisation on data and apply sax with a sliding window
    sax_none, paa_coef = sax_via_window(datas, win_size=window_size, paa_size=paa_size, alphabet_size=alphabet_size, nr_strategy=None, z_threshold=0.01)

    # Split paa_coef into train and test
    paa_coef_train = paa_coef[:231]
    paa_coef_test = paa_coef[231:]

    pd.DataFrame(paa_coef_train).to_csv('data/saxpy_results/paa_coef_training.csv')
    pd.DataFrame(paa_coef_test).to_csv('data/saxpy_results/paa_coef_testing.csv')

    # Create two new dictionaries to store the split data
    sax_none_training_temp = {}
    sax_none_testing_temp = {}
    sax_none_training = {}
    sax_none_testing = {}


    # Iterate over the original data and split based on the condition to distinguish between train and test data
    for key, values in sax_none.items():
        sax_none_training_temp[key] = [value for value in values if value < 22176]
        sax_none_testing_temp[key] = [value for value in values if value >= 22176]

    for key, value in sax_none_training_temp.items():
        if value:
            sax_none_training[key] = value

    for key, value in sax_none_testing_temp.items():
        if value:
            sax_none_testing[key] = value

    key_length_training = sax_none_exploration(sax_none_training)
    key_length_training.to_csv(rf'data/saxpy_results/paa_{paa}_alphabet_{alphabet_size}_training.csv')

    key_length_testing = sax_none_exploration(sax_none_testing)
    key_length_testing.to_csv(rf'data/saxpy_results/paa_{paa}_alphabet_{alphabet_size}_testing.csv')

    df_sax_none_training = pd.DataFrame([sax_none_training]).transpose()
    df_sax_none_training.to_csv(rf'data/saxpy_results/paa_{paa}_alphabet_{alphabet_size}_sax_none_training.csv')
    df_sax_none_testing = pd.DataFrame([sax_none_testing]).transpose()
    df_sax_none_testing.to_csv(rf'data/saxpy_results/paa_{paa}_alphabet_{alphabet_size}_sax_none_testing.csv')


paa_size = 12 #2 hours interval, for 4 -> 6 hours interval
alphabet_size = 3
saxpy(data, paa_size, alphabet_size)

print("Finished saxpy")
