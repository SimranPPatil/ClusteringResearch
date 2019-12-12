from KMeans import *
from DBSCAN import *
from DBSCAN_without_choice import *
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import subprocess
import os

def KMeans_main(maxiter, fraction):
    X, X_population, rnd_indices = KMeans_initialize(fraction)
    k = determine_K(X)
    
    #Defining centers
    centroids = np.random.rand(k,2)
    
    # Initialize the vectors in which we will store the assigned classes of each data point and the calculated distances from each centroid
    classes = np.zeros(X.shape[0], dtype=np.float64)
    distances = np.zeros([X.shape[0], k], dtype=np.float64)

    # start_kmeans = time.time()
    centroids, classes = MyKMeans(maxiter, centroids, classes, distances, X, k)
    # end_kmeans = time.time()
    #print(centroids)

    accuracy_kmeans = validate_Kmeans(X, centroids, classes, rnd_indices)
    
    # Time taken in seconds
    # time_kmeans = end_kmeans - start_kmeans
    # print("Elapsed (after compilation) = %s" % time_kmeans)

    return accuracy_kmeans, k, X_population, X, rnd_indices


def validate_DBSCAN(skl_labels, my_labels):
    # Scikit learn uses -1 to for NOISE, and starts cluster labeling at 0. I start numbering at 1, so increment the skl cluster numbers by 1.
    for i in range(len(skl_labels)):
        if not skl_labels[i] == -1:
            skl_labels[i] += 1

    num_disagree = 0
    # Go through each label and make sure they match (print the labels if they # don't)
    count = 0
    for i in range(len(skl_labels)):
        if not skl_labels[i] == my_labels[i]:
            #print('Scikit learn:', skl_labels[i], 'mine:', my_labels[i])
            num_disagree += 1
        else:
            count += 1
    return num_disagree/len(skl_labels)

def get_labels_DBSCAN(rnd_indices):
	f = open("make_blobs_labels.txt", "r")
	lines = f.readlines()
	classified_labels = []

	#Extracting labels from the file
	for i in rnd_indices:
		classified_labels.append(int(lines[i].strip("\n")))

	return classified_labels


def DBSCAN_main(X, rnd_indices):
    X = StandardScaler().fit_transform(X)

    # start_dbscan = time.time()
    dictionary = {"if_mydbscan":0, "else_mydbscan":0, "else_gc_body_subordinate_if":0, "else_gc":0}
    my_labels, dictionary = MyDBSCAN(X, dictionary, eps=0.3, MinPts=10)
    # end_dbscan = time.time()

    # time_dbscan = end_dbscan - start_dbscan
    # print("Elapsed (after compilation) = %s" % (time_dbscan))

    skl_labels = get_labels_DBSCAN(rnd_indices)
    num_disagree = validate_DBSCAN(skl_labels, my_labels) 
    return num_disagree, dictionary

def dbscan_cost(N, dictionary):
    sub_cost = 1
    add_cost = 1
    mul_cost = 2
    div_cost = 2
    comp_lt_cost = 1
    comp_eq_cost = 1
    comp_get_cost = 1
    comp_neq_cost = 1
    comp_gteq_cost = 1

    region_query_cost = N * (sub_cost+mul_cost+comp_lt_cost)
    else_gc_body_cost = comp_eq_cost + region_query_cost + comp_gteq_cost + 0 + (dictionary["else_gc_body_subordinate_if"] * add_cost)
    grow_cluster_cost = N * ((comp_lt_cost + comp_eq_cost + add_cost) + 0 + (dictionary["else_gc"] * else_gc_body_cost))
    dbscsn_cost = N * (comp_neq_cost + region_query_cost + 
    ( (dictionary["if_mydbscan"] * (comp_eq_cost+comp_lt_cost)) + (dictionary["else_mydbscan"] * (add_cost+comp_eq_cost+comp_lt_cost+grow_cluster_cost)) ) )

    return dbscsn_cost

def kmeans_cost(N):
    # we define costs for certain operations so that it will eventually contribute to the net cost computation
        # N is determined at run time so it is passed as a parameter
    sub_cost = 1
    add_cost = 1
    mul_cost = 2
    div_cost = 2

    kmeans_cost = N * (maxiter * k * (2 * sub_cost + add_cost + 2 * mul_cost) + 1 + k)
    return kmeans_cost

def build_stats_dict(filename):
        cache_stats = {}
        with open(filename, "r") as input_file:
            for line in input_file:
                try:
                    data = line.split("== ")[1]
                    data = data.strip().split(":")
                    if len(data[0]) > 0:
                        key = data[0]
                        val = data[1].strip().split(" ")[0]
                        cache_stats.setdefault(key, val)
                except Exception as e:
                    print(e)
        for key in cache_stats:
            print(key)
        return cache_stats

def cost_analysis(length, dictionary, k, maxiter, fraction):
    KMeans_cost = kmeans_cost(length)
    print("KMeans cost = ", KMeans_cost)

    DBSCAN_cost = dbscan_cost(length, dictionary)
    print("DBSCAN_cost = ", DBSCAN_cost)
    
    # getting memory costs for kmeans
    cmd = "valgrind --tool=cachegrind python3 KMeans.py " + str(k) + " " + str(maxiter) + " " + str(fraction)
    with open("temp", "w") as f:
        subprocess.Popen(cmd, stdout=f, stderr=f, shell=True).wait()

    cmd = "tail -n16 temp"
    with open("input_kmeans", "w") as f:
        subprocess.Popen(cmd, stdout=f, stderr=f, shell=True).wait()

    # getting memory costs for dbscan
    cmd = "valgrind --tool=cachegrind python3 DBSCAN.py " + str(fraction)
    with open("temp", "w") as f:
        subprocess.Popen(cmd, stdout=f, stderr=f, shell=True).wait()

    cmd = "tail -n16 temp"
    with open("input_dbscan", "w") as f:
        subprocess.Popen(cmd, stdout=f, stderr=f, shell=True).wait()
     
    cache_stats_kmeans = build_stats_dict("input_kmeans")
    print(cache_stats_kmeans)

    Memory_cost = (cache_stats_kmeans['I   refs'] + cache_stats_kmeans['D   refs'] )*0.206864 + cache_stats_kmeans['LL refs']*4.34186 + cache_stats_kmeans['LL misses']*(0.02+0.282094)

    cache_stats_dbscan = build_stats_dict("input_dbscan")
    print(cache_stats_dbscan)

    KMeans_cost += KMeans_cost + Memory_cost
    DBSCAN_cost += DBSCAN_cost + Memory_cost
    print(KMeans_cost + DBSCAN_cost)
        
    return KMeans_cost, DBSCAN_cost, cache_stats_kmeans, cache_stats_dbscan

def final_choice(KMeans_cost, DBSCAN_cost, X_population, k, maxiter):
    X= X_population
    if(KMeans_cost > DBSCAN_cost):
        X = StandardScaler().fit_transform(X)
        my_labels = MyDBSCAN(X, eps=0.3, MinPts=10)
    else:
                
        classes = np.zeros(X.shape[0], dtype=np.float64)
        distances = np.zeros([X.shape[0], k], dtype=np.float64)
        centroids = np.random.rand(k,2) #KMeans
                
        centroids, classes = MyKMeans(maxiter, centroids, classes, distances, X_population, k)

if __name__ == "__main__":
   
    if len(sys.argv) < 2:
        print("Enter fraction")
        exit()

    fraction = float(sys.argv[1])
    maxiter = 50
    accuracy_kmeans, k, X_population, X, rnd_indices = KMeans_main(maxiter, fraction)
    data_size = int(len(X_population) * fraction)
    print("Accuracy_KMEANS", accuracy_kmeans)
    if accuracy_kmeans == 1:
	    X= X_population
	    centroids = np.random.rand(k,2) #KMeans
	    classes = np.zeros(X.shape[0], dtype=np.float64)
	    distances = np.zeros([X.shape[0], k], dtype=np.float64)
		
	    centroids, classes = MyKMeans(maxiter, centroids, classes, distances, X, k)
	    print(centroids)
    else:
		#DBSCAN
        num_disagree, dictionary = DBSCAN_main(X, rnd_indices)
        print("Accuracy_DBSCAN", 1-num_disagree)

        if num_disagree == 0:
            print('PASS - All labels match!')

            X= X_population
            X = StandardScaler().fit_transform(X)
            my_labels = MyDBSCAN_without_cost(X, eps=0.3, MinPts=10)

        else:
            print('FAIL -', num_disagree, 'labels don\'t match.')
            KMeans_cost, DBSCAN_cost, cache_stats_kmeans, cache_stats_dbscan = cost_analysis(len(X), dictionary, k, maxiter, fraction)
            final_choice(KMeans_cost, DBSCAN_cost, X_population, k, maxiter)
