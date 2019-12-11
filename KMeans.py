import sys
import time
import subprocess
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def KMeans_initialize():   
    X_population = []
    X_temp = []

    #data cleaning and reading from the input file
    with open("make_blobs_data.txt", "r") as f:
        for line in f:
            line = line.strip().split(" ")
            X_values = []
            X_values.append(float(line[0]))
            X_values.append(float(line[1]))
            X_temp.append(X_values)
    
    X_population = np.array(X_temp) #Losing precision
    indices = np.arange(len(X_population))

    #Selects the number of data points from the dataset
    rnd_indices = np.random.choice(indices, size=300)
    X = X_population[rnd_indices]
    return X, X_population

def determine_K(X):
    #Determining the value of k in kmeans
    distortions = []
    for k in range(1,10):
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    
    #Using Elbow Method to determine the value of k
    k = 0
    while(distortions[k] - distortions[k+1] > 0.1):
        k = k+1
    k = k+1
    return k

def get_distances(centroid, points):
    """Returns the distance the centroid is from each data point in points."""
    return np.linalg.norm(points - centroid, axis=1)

def MyKMeans(maxiter, centroids, classes, distances, X, k):
    # Loop for the maximum number of iterations
    for iteration in range(maxiter):
        # Assign all points to the nearest centroid
        for num, value in enumerate(centroids):
            distances[:, num] = get_distances(value, X)
        
        # Determine class membership of each point by picking the closest centroid
        classes = np.argmin(distances, axis=1)
    
        # Update centroid location using the newly assigned data point classes
        for centroid in range(k):
            centroids[centroid] = np.mean(X[classes == centroid], 0)

    return centroids, classes

def validate_Kmeans(X, centroids):
	#Validating
	kmeans = KMeans(n_clusters=3)
	kmeans = kmeans.fit(X)
	centroids_orig = kmeans.cluster_centers_

	iteration = 0
	#Assuming 100% accuracy
	accuracy_kmeans = 1
	while(centroids[iteration] not in centroids_orig and accuracy_kmeans != 0 and iteration < 3):
		accuracy_kmeans = 0 #Accuracy not 100%
		iteration += 1
	#print(accuracy_kmeans)
	return accuracy_kmeans


if __name__ == "__main__":
	if len(sys.argv) < 3:
		exit()

	k =  int(sys.argv[1])
	maxiter = int(sys.argv[2])
	print(k,maxiter)

	X, X_population = KMeans_initialize()

	centroids = np.asarray([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]) #KMeans
	classes = np.zeros(X.shape[0], dtype=np.float64)
	distances = np.zeros([X.shape[0], k], dtype=np.float64)
        
	centroids, classes = MyKMeans(maxiter, centroids, classes, distances, X, k)
	print(centroids)
	

	
