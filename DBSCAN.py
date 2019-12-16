import sys
import time
import subprocess
import numpy as np
import numpy
from numba import jit
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def MyDBSCAN(D, dictionary, eps, MinPts):
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
            dictionary["if_mydbscan"] += 1
        else: 
            C += 1
            dictionary["else_mydbscan"] += 1
            dictionary = growCluster(D, labels, P, NeighborPts, C, eps, MinPts, dictionary)

    # All data has been clustered!
    return labels, dictionary

def growCluster(D, labels, P, NeighborPts, C, eps, MinPts, dictionary):
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
            dictionary["else_gc"] += 1
            labels[Pn] = C
            PnNeighborPts = regionQuery(D, Pn, eps)
            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts
                dictionary["else_gc_body_subordinate_if"] += 1
        i += 1  
        
    return dictionary

def regionQuery(D, P, eps):
    neighbors = []
    # For each point in the dataset...
    for Pn in range(0, len(D)):
        # If the distance is below the threshold, add it to the neighbors list.
        if numpy.linalg.norm(D[P] - D[Pn]) < eps:
            neighbors.append(Pn)
            
    return neighbors



