import matplotlib.pylab as plt
import matplotlib.patches as mpatches
import pandas as pd

def plot_features(data):
    plt.figure()
    plt.rcParams.update({'font.size': 16})
    data.plot()
    #plt.title("Features")
    plt.xlabel('Observations')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

def plot_labels(labels):
    plt.plot(labels)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
def plot_colored_line_on_labels(x, y, desc):
    x1 = np.linspace(0, 3 * np.pi, 500)
    #y = np.sin(x)
    dydx = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative

    # Create a set of line segments so that we can color them individually
    # This creates the points as an N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    #points = np.array([x, y]).T.reshape(-1, 1, 2)
    #segments = np.concatenate([points[:-1], points[1:]], axis=1)

    points = np.array([x, y])
    segments = np.concatenate([points[:-1], points[1:]])

    fig, axs = plt.subplots()

    axs.set_xlim(x.min(), x.max())
    axs.set_ylim(y.min(), y.max())

    # Create a continuous norm to map from data points to colors
    # norm = plt.Normalize(dydx.min(), dydx.max())
    # lc = LineCollection(segments, cmap='viridis', norm=norm)
    # # Set the values used for colormapping
    # lc.set_array(dydx)
    # lc.set_linewidth(2)
    # line = axs[0].add_collection(lc)
    # fig.colorbar(line, ax=axs[0])

    # Use a boundary norm instead
    cmap = ListedColormap(['g', 'b'])
    norm = BoundaryNorm([-1, 0, 1], cmap.N)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(desc)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)
    fig.colorbar(line)


    plt.show()
    print("Finished")

def plot_cluster_centers(centroids):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})

    plt.ylabel("$\\Delta$ Temperature (Normalized)") #Normalized
    plt.xlabel("Time [Hours]")
    plt.plot(centroids[2, 0])
    plt.plot(centroids[1, 0])
    plt.plot(centroids[0, 0])
    # plt.plot(centroids[1, 0])
    # plt.plot(centroids[0, 0])
    # plt.plot(centroids[2, 0])

    # Create a list of x-tick labels representing every 6 hours of the day
    xticks = [f'{i}:00' for i in range(0, 24, 6)]
    # Set the x-tick labels to the custom list of hour labels
    ax.set_xticks(range(0, len(centroids.transpose()), int(len(centroids.transpose()) / 4)))
    ax.set_xticklabels(xticks)
    #plt.savefig('data/graphics/centroids_kmeans')
    plt.savefig('data/graphics/centroids_sax_kmeans')
    plt.show()

def plot_clustered_data(data_labeled):
    t = data_labeled[:,2].astype(int)
    x = data_labeled[:,0]
    s = data_labeled[:,1]

    cluster_3 = np.where(t == 2, s, None)
    cluster_2 = np.where(t == 1, s, None)
    cluster_1 = np.where(t == 0, s, None)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})
    plt.ylabel("$\\Delta$ Temperature [K]")
    plt.xlabel("Timestamp")
    #ax.plot(x, cluster_1, x, cluster_2, x, cluster_3) #kmeans
    #ax.plot(x, cluster_3, x, cluster_2, x, cluster_1) #sax
    #ax.plot(x, cluster_1, x, cluster_3, x, cluster_2)
    ax.plot(x, cluster_2, x, cluster_1, x, cluster_3) #feature
    #plt.savefig('data/graphics/labeled_data_test')
    #plt.savefig('data/graphics/clustered_data_sax_kmeans')
    plt.show()

def plot_clustered_data_with_days(data_labeled):
    t = data_labeled[:,2].astype(int)
    #x = data_labeled[:,0].astype(int)
    x = np.arange(0,len(data_labeled))
    s = data_labeled[:,1]

    cluster_4 = np.where(t == 3, s, None)
    cluster_3 = np.where(t == 2, s, None)
    cluster_2 = np.where(t == 1, s, None)
    cluster_1 = np.where(t == 0, s, None)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})
    #plt.title("Clustered Data")
    plt.ylabel("$\\Delta$ Temperature [K]")
    plt.xlabel("Indices")
    ax.plot(x, cluster_1, x, cluster_2, x, cluster_3, x, cluster_4)
    for day in range(0, int(len(s)/96)):
        plt.axvline(x=day * 96, color='0.6', linestyle='--', linewidth = 1) #, label=day) #, alpha=0.5)
        #plt.legend()
    plt.show()

def plot_clustered_data_with_anomalies(data_labeled, anomalie_indexes_1, anomalie_indexes_2):
    t = data_labeled[:,2].astype(int)
    x = data_labeled[:,0]
    #x1 = np.arange(0, len(data_labeled))
    s = data_labeled[:,1]

    cluster_3 = np.where(t == 2, s, None)
    cluster_2 = np.where(t == 1, s, None)
    cluster_1 = np.where(t == 0, s, None)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})
    plt.ylabel("$\\Delta$ Temperature [K]")
    plt.xlabel("Timestamp")
    #ax.plot(x, cluster_1, x, cluster_2, x, cluster_3) #sax
    #ax.plot(x, cluster_2, x, cluster_1, x, cluster_3) #feature
    ax.plot(x, cluster_3, x, cluster_2, x, cluster_1) #kmeans

    # Plot week lines
    for week in range(0, int(len(s)/(96*7)) + 1):
        plt.axvline(x=x[week * (96*7)], color='0.6', linestyle='--', linewidth = 1)

    # Plot anomalies
    anomalie_indexes_1['0'] = pd.to_datetime(anomalie_indexes_1['0'])
    delta = pd.Timedelta(days=7)
    for i in range(0, len(anomalie_indexes_1)):
        for j in range(0, len(x)):
            if anomalie_indexes_1['0'][i] == x[j]:
                print(x[j])
                plt.axvspan(x[j], x[j] + delta, color='purple', alpha=0.4)
    anomalie_indexes_2['0'] = pd.to_datetime(anomalie_indexes_2['0'])
    delta = pd.Timedelta(days=1)
    for i in range(0, len(anomalie_indexes_2)):
        for j in range(0, len(x)):
            if anomalie_indexes_2['0'][i] == x[j]:
                print(x[j])
                plt.axvspan(x[j], x[j] + delta, color='olive', alpha=0.5)

    color_patch_1 = mpatches.Patch(color='purple', alpha=0.4, label='Unusual pattern of valid clusters')
    color_patch_2 = mpatches.Patch(color='olive', alpha=0.5, label='Unusual pattern for cluster')
    # Add the legend to the plot with the color patches
    plt.legend(handles=[color_patch_1, color_patch_2], title='Types of anomalies')
    #plt.savefig('data/graphics/anomalies_feature')
    #plt.savefig('data/graphics/anomalies_sax_kmeans')
    plt.show()

print("plotted")
