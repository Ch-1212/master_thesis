import pandas as pd
import numpy as np
import mergeDF as mDF
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import util_plots as uplt

# Read data
df1 = mDF.read_data_from_Friedrichshagencsv()
df_new_data = mDF.read_new_data_from_Friedrichshagencsv() #1.1.23 (So) - 16.05.23
# Cut new data to fit to old data (delete values that are doubled)
df_new_data = df_new_data[83170:]
df = pd.concat([df1, df_new_data], ignore_index = True)

def preprocess(data):
    # Abstract room temperature
    data[3] = data[1] - data[2]

    def plot_original_data(df):
        plt.figure(figsize=(10,6))
        plt.rcParams.update({'font.size': 14})
        plt.plot(df['Timestamp'],
                 df[1])
        plt.plot(df['Timestamp'],
                 df[2])
        plt.xlabel('Timestamp')
        plt.ylabel('Temperature [°C]')
        plt.savefig('data/graphics/original_data')
        plt.show()

    def plot_substracted_data(df):
        plt.figure(figsize=(10,6))
        plt.rcParams.update({'font.size': 14})
        plt.plot(df['Timestamp'],
                 df[3])
        plt.xlabel('Timestamp')
        plt.ylabel('Temperature [Kelvin]')
        plt.savefig('data/graphics/subtracted_data')
        plt.show()

    plot_original_data(data)
    #plot_substracted_data(data)

    # Convert minute data to 15 min intervals // Alternative siehe unten
    df = data.groupby(
        pd.Grouper(key='Timestamp', freq='15min', origin='start')
    ).mean()

    # Eliminate NAN values through interpolation
    #count_nan = df.isna().sum()  # count_nan = 169 -> 169/38782= 0.004357 ~ 0.44%
    df = df.interpolate()
    # Alternative: x_ = np.nan_to_num(x_) #, nan=1.7)

    return df

df = preprocess(df)

# To make sliding window start at 00:00h, I lose first 12 hours; It now starts at Sunday 2022-04-08 00:07h
df = df[47:]

def plot_cleaned_data():
    plt.figure(figsize=(10,6))
    plt.rcParams.update({'font.size': 14})
    plt.plot(df[3])
    plt.axhline(0, color='0.6', linestyle='--', zorder=-1)
    plt.xlabel('Timestamp')
    plt.ylabel('$\\Delta$ Temperature [K]')
    plt.savefig('data/graphics/cleaned_data')
    plt.show()

plot_cleaned_data()

# Divide data into X_train and X_test (distinction between fit and predict)
X_train, X_test = train_test_split(df, train_size=22176, shuffle=False)
# Test starts at Fri, 25.11. 00:07 Uhr; 43% Test vs 57% Training data

# Export preprocessed data to csv to use further
df.to_csv(r'data/data.csv')
X_train.to_csv(r'data/data_training.csv')
X_test.to_csv(r'data/data_testing.csv')

# Define own_labels for data
def daily_to_data_labels(labels, win_size):
    data_labels = []
    for i in range(len(labels)):
        for j in range(win_size):
            data_labels.append(labels[i])
    return pd.Series(data_labels)

own_daily_labels = pd.read_csv('data/own_labels_daily.csv')
own_labels = daily_to_data_labels(own_daily_labels['own_labels'], 96)
data_labels = np.stack((df[3][:len(own_labels)], own_labels), axis=1)
data_labels = pd.DataFrame(data_labels, index=df.index[:len(own_labels)], columns=['data', 'own_labels'])

# Find out cluster apportionment
#data_labels['own_labels'].value_counts()[0]

# Convert the index and the columns to ndarrays with the object data type, to enable the plotting
index_arr = np.array(data_labels.index, dtype=object)[:, np.newaxis]
cols_arr = np.array(data_labels, dtype=object)

#uplt.plot_clustered_data(np.column_stack((index_arr, cols_arr)))
data_labels.to_csv('data/own_labels.csv')

# Split data into test and training data
own_labels_train, own_labels_test = train_test_split(data_labels, train_size=len(X_train), shuffle=False)
own_labels_train.to_csv(r'data/own_labels_training.csv')
own_labels_test.to_csv(r'data/own_labels_testing.csv')

# Convert the index and the columns to ndarrays with the object data type, to enable the plotting of the clustered test data
index_arr_test = np.array(own_labels_test.index, dtype=object)[:, np.newaxis]
cols_arr_test = np.array(own_labels_test, dtype=object)

uplt.plot_clustered_data(np.column_stack((index_arr_test, cols_arr_test)))

# Statistical approach for labeling: Divide into labels (Zapfen(1); Nicht-Zapfen(0))
# def create_labels(index, threshold):
#     label = np.zeros(len(df))
#     for i in range(0, len(df)):
#         if df[index][i] < threshold:
#             label[i] = 1
#     df_label = pd.DataFrame(label.transpose(), dtype=int)
#     #df[5] = df_label
#     df[5] = label.transpose()
#     df[5].astype(dtype=int) #no effect
#
#
# # Use original data
# #create_labels(1, 20)
#
# # Use Gradients
# #create_labels(4, 0)
#
# def plot_labeled_data():
#     t = np.array(df[5])
#     x = np.arange(1, len(df) + 1)
#     s = np.array(df[3])
#
#     supper = np.where(t == 1, s, None)
#     slower = np.where(t == 0, s, None)
#
#     fig, ax = plt.subplots()
#     # ax.title("Labeled Data")
#     ax.plot(x, slower, x, supper)
#     plt.show()

#plot_labeled_data()


# As you stated that you have data for every minute, i.e. there's always the same number of values per day (1440), then you can simply take the daily means and then the means of a 5-day rolling window on these daily means.
#
# Example (value is running minute number in year, starting with 0):
#
# s = pd.Series(pd.date_range('2018-01-01', '2018-12-31 23:59', freq='1T'))
# df = pd.DataFrame(s.index.values, index=s, columns=['Value'])
# df.groupby(df.index.floor('d'))['Value'].mean().rolling(5).mean().dropna()