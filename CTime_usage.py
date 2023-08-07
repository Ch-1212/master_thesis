#import timeit, cProfile, BigO sind ungeeignet
import time
import pandas as pd

# Execution time results in seconds
def cpu_result():
    cpu_results_c = [9.810509, 11.301867, 2.320465]
    cpu_results_tr = []
    cpu_results_te = [5.426114, 0.789476, 1.137027]
    cpu_results_without_anomalies = [5.405328, 0.582342, 1.106024]
    cpu_results = pd.DataFrame([cpu_results_c, cpu_results_te]).transpose()
    cpu_results.columns = ['Complete process', 'Testing process']
    cpu_results.to_csv(r'data/results/ctime_results.csv')

cpu_result()

# Runtime with time; import only works at first time used
sax_k = []
kmeans = []
feature = []

def time_complete():
    start_time = time.time()

    def sax():
        import Saxpy
        import Sax_cluster
        import Sax_cluster_testing

    sax()

    end_time = time.time()
    elapsed_time_sax = end_time - start_time
    sax_k.append(elapsed_time_sax)
    print(f'Elapsed time for SAX: {elapsed_time_sax} seconds')

    start_time = time.time()

    def ts_kmeans():
        import Kmeans
        import Kmeans_testing

    ts_kmeans()

    end_time = time.time()
    elapsed_time_kmeans = end_time - start_time
    kmeans.append(elapsed_time_kmeans)
    print(f'Elapsed time for Kmeans: {elapsed_time_kmeans} seconds')

    start_time = time.time()

    def fe():
        import Feature_kmeans
        import Feature_kmeans_testing

    fe()

    end_time = time.time()
    elapsed_time_fe = end_time - start_time
    feature.append(elapsed_time_fe)
    print(f'Elapsed time Feature: {elapsed_time_fe} seconds')

    cpu_results = pd.DataFrame([sax_k, kmeans, feature]).transpose()
    cpu_results.columns = ['SAX', 'K-means', 'Feature K-means']
    # SAX    K-means  Feature K-means [0  9.810509  11.301867         2.320465]

def time_training():
    start_time = time.time()

    import Saxpy
    import Sax_cluster

    end_time = time.time()
    elapsed_time_sax = end_time - start_time
    sax_k.append(elapsed_time_sax)
    print(f'Elapsed time for SAX: {elapsed_time_sax} seconds')

    start_time = time.time()

    import Kmeans

    end_time = time.time()
    elapsed_time_kmeans = end_time - start_time
    kmeans.append(elapsed_time_kmeans)
    print(f'Elapsed time for Kmeans: {elapsed_time_kmeans} seconds')

    start_time = time.time()

    import Feature_kmeans

    end_time = time.time()
    elapsed_time_fe = end_time - start_time
    feature.append(elapsed_time_fe)
    print(f'Elapsed time Feature: {elapsed_time_fe} seconds')

    cpu_results = pd.DataFrame([sax_k, kmeans, feature]).transpose()
    cpu_results.columns = ['SAX', 'K-means', 'Feature K-means']
    # 0          1        2 [8.401202  13.675229  1.45533]

def time_testing():
    # Compare only testing time
    start_time = time.time()

    import Sax_cluster_testing

    end_time = time.time()
    elapsed_time_sax_testing = end_time - start_time
    sax_k.append(elapsed_time_sax_testing)
    print(f'Elapsed time for SAX: {elapsed_time_sax_testing} seconds')

    start_time = time.time()

    import Kmeans_testing

    end_time = time.time()
    elapsed_time_kmeans_testing = end_time - start_time
    kmeans.append(elapsed_time_kmeans_testing)
    print(f'Elapsed time for Kmeans: {elapsed_time_kmeans_testing} seconds')

    start_time = time.time()

    import Feature_kmeans_testing

    end_time = time.time()
    elapsed_time_fe_testing = end_time - start_time
    feature.append(elapsed_time_fe_testing)
    print(f'Elapsed time Feature: {elapsed_time_fe_testing} seconds')

    cpu_results = pd.DataFrame([sax_k, kmeans, feature]).transpose()
    cpu_results.columns = ['SAX', 'K-means', 'Feature K-means']
    # [5.426114, 0.789476, 1.137027]

#time_complete()
#time_training()
#time_testing()