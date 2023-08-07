import tracemalloc
import pandas as pd

# Memory results in MB
def cpu_result():
    cpu_results_c = [73.716775, 8.893931, 4.395026]
    cpu_results_te = [69.426035, 2.336609, 2.951791]
    cpu_results = pd.DataFrame([cpu_results_c, cpu_results_te]).transpose()
    cpu_results.columns = ['Complete process', 'Testing process']
    cpu_results.to_csv(r'data/results/mem_results.csv')

cpu_result()

# Start tracing memory allocations
tracemalloc.start()

def mem_complete():
    # Get current memory usage
    current1, peak1 = tracemalloc.get_traced_memory()

    # Call the function
    import Saxpy
    import Sax_cluster
    import Sax_cluster_testing

    # Get new memory usage
    current2, peak2 = tracemalloc.get_traced_memory()

    # Calculate the difference in memory usage
    mem_diff_sax = (current2 - current1) / 1024 / 1024
    print(f'Memory usage difference: {mem_diff_sax} MB')

    current1, peak1 = tracemalloc.get_traced_memory()

    import Kmeans
    import Kmeans_testing

    current2, peak2 = tracemalloc.get_traced_memory()

    mem_diff_kmeans = (current2 - current1) / 1024 / 1024
    print(f'Memory usage difference: {mem_diff_kmeans} MB')

    current1, peak1 = tracemalloc.get_traced_memory()

    import Feature_kmeans
    import Feature_kmeans_testing

    current2, peak2 = tracemalloc.get_traced_memory()

    mem_diff_fe = (current2 - current1) / 1024 / 1024
    print(f'Memory usage difference: {mem_diff_fe} MB')

    cpu_results = pd.DataFrame([mem_diff_sax, mem_diff_kmeans, mem_diff_fe]).transpose()
    cpu_results.columns = ['SAX', 'K-means', 'Feature K-means']
    # SAX   K-means  Feature K-means [0  73.716775  8.893931 4.395026]

def mem_testing():
    current1, peak1 = tracemalloc.get_traced_memory()

    import Sax_cluster_testing

    current2, peak2 = tracemalloc.get_traced_memory()

    mem_diff_sax = (current2 - current1) / 1024 / 1024
    print(f'Memory usage difference: {mem_diff_sax} MB')

    current1, peak1 = tracemalloc.get_traced_memory()

    import Kmeans_testing

    current2, peak2 = tracemalloc.get_traced_memory()

    mem_diff_kmeans = (current2 - current1) / 1024 / 1024
    print(f'Memory usage difference: {mem_diff_kmeans} MB')

    current1, peak1 = tracemalloc.get_traced_memory()

    import Feature_kmeans_testing

    current2, peak2 = tracemalloc.get_traced_memory()

    mem_diff_fe = (current2 - current1) / 1024 / 1024
    print(f'Memory usage difference: {mem_diff_fe} MB')

    cpu_results = pd.DataFrame([mem_diff_sax, mem_diff_kmeans, mem_diff_fe]).transpose()
    cpu_results.columns = ['SAX', 'K-means', 'Feature K-means']
    # SAX   K-means  Feature K-means [0  69.426035  2.336609  2.951791]

#mem_complete()
#mem_testing()

# Stop tracing memory allocations
tracemalloc.stop()