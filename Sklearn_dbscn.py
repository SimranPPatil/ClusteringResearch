from KMeans import *
from DBSCAN import *
from DBSCAN_without_choice import *
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import subprocess
import os
from decimal import Decimal
import time

def DBSCAN_initialize(fraction, data_file):   
    X_population = []
    X_temp = []

    #data cleaning and reading from the input file
    with open(data_file, "r") as f:
        for line in f:
            line = line.strip().split(" ")
            X_values = []
            X_values.append(float(line[0]))
            X_values.append(float(line[1]))
            X_temp.append(X_values)
    
    X_population = np.array(X_temp) #Losing precision
    #indices = np.arange(len(X_population))
    data_size = int(len(X_population)*fraction)
    #print("Current data size: ", data_size)
    #Selects the number of data points from the dataset
    #rnd_indices = np.random.choice(indices, size=data_size)
    #X = X_population[rnd_indices]
    X = X_population[:data_size]
    return X, X_population

if __name__ == "__main__":
	
	data_file = sys.argv[1]
	fraction = 1.0	

	start_kmeans = time.time()

	X, X_population = DBSCAN_initialize(fraction, data_file)

	
	X = StandardScaler().fit_transform(X)
	db = DBSCAN(eps=0.3, min_samples=10).fit(X)

	end_kmeans = time.time()

	#Time taken in seconds
	time_kmeans = end_kmeans - start_kmeans
	print("Elapsed (after compilation) = %s" % time_kmeans)
