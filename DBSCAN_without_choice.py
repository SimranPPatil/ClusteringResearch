import sys
import time
import subprocess
import numpy as np
import numpy
from numba import jit
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def DBSCAN_initialize(fraction):   
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
    data_size = int(len(X_population) * fraction)
    #Selects the number of data points from the dataset
    rnd_indices = np.random.choice(indices, size=data_size)
    X = X_population[rnd_indices]
    return X, X_population

def MyDBSCAN_without_cost(D, eps, MinPts):
    # -1 - Indicates a noise point; 0 - Means the point hasn't been considered yet.
    # Initially all labels are 0.    
    labels = [0]*len(D)
    # C is the ID of the current cluster.    
    C = 0
    for P in range(0, len(D)):
        if not (labels[P] == 0):
            continue
        # Find all of P's neighboring points.
        NeighborPts = regionQuery(D, P, eps)
        if len(NeighborPts) < MinPts:
            labels[P] = -1
        else: 
            C += 1
            growCluster(D, labels, P, NeighborPts, C, eps, MinPts)
    # All data has been clustered!
    return labels

def growCluster(D, labels, P, NeighborPts, C, eps, MinPts):
    #`C` - The label for this new cluster.  
    # Assign the cluster label to the seed point.
    labels[P] = C
    i = 0
    while i < len(NeighborPts):    
        # Get the next point from the queue.        
        Pn = NeighborPts[i]
        if labels[Pn] == -1:
            labels[Pn] = C
        elif labels[Pn] == 0:
            labels[Pn] = C
            PnNeighborPts = regionQuery(D, Pn, eps)
            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts
        i += 1  

def regionQuery(D, P, eps):
    neighbors = []
    for Pn in range(0, len(D)):
        # If the distance is below the threshold, add it to the neighbors list.
        if numpy.linalg.norm(D[P] - D[Pn]) < eps:
            neighbors.append(Pn)
    return neighbors


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Enter fraction of dataset")
        exit()
        
    fraction = float(sys.argv[1])
    X, X_population = DBSCAN_initialize(fraction)
    my_labels = MyDBSCAN(X, eps=0.3, MinPts=10)
	
