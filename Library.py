from KMeans import *
from DBSCAN import *
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import subprocess
import os

def KMeans_main(maxiter):
    X, X_population = KMeans_initialize()
    k = determine_K(X)
    
    #Defining centers
    centroids = np.asarray([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
    
    # Initialize the vectors in which we will store the assigned classes of each data point and the calculated distances from each centroid
    classes = np.zeros(X.shape[0], dtype=np.float64)
    distances = np.zeros([X.shape[0], k], dtype=np.float64)
    centroids, classes = MyKMeans(maxiter, centroids, classes, distances, X, k)
    accuracy_kmeans = validate_Kmeans(X, centroids)
    
    return accuracy_kmeans, k, X_population, X


def validate_DBSCAN(skl_labels, my_labels):
    # Scikit learn uses -1 to for NOISE, and starts cluster labeling at 0. I start numbering at 1, so increment the skl cluster numbers by 1.
    for i in range(0, len(skl_labels)):
        if not skl_labels[i] == -1:
            skl_labels[i] += 1

    num_disagree = 0
    # Go through each label and make sure they match (print the labels if they # don't)
    count = 0
    for i in range(0, len(skl_labels)):
        if not skl_labels[i] == my_labels[i]:
            print('Scikit learn:', skl_labels[i], 'mine:', my_labels[i])
            num_disagree += 1
        else:
            count = count+1
    return num_disagree

def DBSCAN_main(X):
    X = StandardScaler().fit_transform(X)
    dictionary = {"if_mydbscan":0, "else_mydbscan":0, "else_gc_body_subordinate_if":0, "else_gc":0}
    my_labels, dictionary = MyDBSCAN(X, dictionary, eps=0.3, MinPts=10)
    
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    skl_labels = db.labels_
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
        return cache_stats

def cost_analysis(length, dictionary, k, maxiter):
    KMeans_cost = kmeans_cost(length)
    print("KMeans cost = ", KMeans_cost)

    DBSCAN_cost = dbscan_cost(length, dictionary)
    print("DBSCAN_cost = ", DBSCAN_cost)
    
    # getting memory costs for kmeans
    cmd = "valgrind --tool=cachegrind python3 KMeans.py " + str(k) + " " + str(maxiter)
    with open("temp", "w") as f:
        subprocess.Popen(cmd, stdout=f, stderr=f, shell=True).wait()
    cmd = "tail -n16 temp>&input_kmeans"
    subprocess.Popen(cmd, shell=True).wait()

    # getting memory costs for dbscan
    cmd = "valgrind --tool=cachegrind python3 DBSCAN.py"
    with open("temp", "w") as f:
        subprocess.Popen(cmd, stdout=f, stderr=f, shell=True).wait()
    cmd = "tail -n16 temp>&input_dbscan"
    subprocess.Popen(cmd, shell=True).wait()
     
    cache_stats_kmeans = build_stats_dict("input_kmeans")
    print(cache_stats_kmeans)
    cache_stats_dbscan = build_stats_dict("input_dbscan")
    print(cache_stats_dbscan)
        
    return KMeans_cost, DBSCAN_cost, cache_stats_kmeans, cache_stats_dbscan

def final_choice(KMeans_cost, DBSCAN_cost, X_population, k, maxiter):
    X= X_population
    if(KMeans_cost > DBSCAN_cost):
        X = StandardScaler().fit_transform(X)
        my_labels = MyDBSCAN(X, eps=0.3, MinPts=10)
    else:
                
        classes = np.zeros(X.shape[0], dtype=np.float64)
        distances = np.zeros([X.shape[0], k], dtype=np.float64)
        centroids = np.asarray([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]) #KMeans
                
        centroids, classes = MyKMeans(maxiter, centroids, classes, distances, X_population, k)

if __name__ == "__main__":
   
    maxiter = 50
    accuracy_kmeans, k, X_population, X = KMeans_main(maxiter)
    
    # if accuracy_kmeans == 1:
    #     X= X_population
    #     centroids = np.asarray([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]) #KMeans
    #     classes = np.zeros(X.shape[0], dtype=np.float64)
    #     distances = np.zeros([X.shape[0], k], dtype=np.float64)
        
    #     centroids, classes = MyKMeans(maxiter, centroids, classes, distances, X, k)
    #     print(centroids)
    # else:
    #     #DBSCAN
    num_disagree, dictionary = DBSCAN_main(X)
    print('FAIL -', num_disagree, 'labels don\'t match.')
    KMeans_cost, DBSCAN_cost, cache_stats_kmeans, cache_stats_dbscan = cost_analysis(len(X), dictionary, k, maxiter)
    final_choice(KMeans_cost, DBSCAN_cost, X_population, k, maxiter)
