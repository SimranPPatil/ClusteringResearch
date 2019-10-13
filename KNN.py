
# coding: utf-8

# In[9]:


import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import operator
from scipy.special import expit
from numba import jit, prange
import os
import time
get_ipython().magic('matplotlib inline')


# In[10]:


centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=1500, centers=centers, cluster_std=0.4, random_state=0)
#print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, labels_true, test_size=0.33, random_state=42)


# f_x_train = np.empty([])
# f_x_test = np.empty([])

# if os.path.exists("train.txt"):
#     os.remove("train.txt")
# if os.path.exists("text.txt"):
#     os.remove("test.txt")

# f = open("train.txt", "a+")
# f.write(str(len(X_train)))
# f.write(' ')
# f.write(str(len(X_train[0])))
# f.write('\n')
# for i in range(0, len(X_train)):
#     np.savetxt(f, X_train[i], fmt='%1.8f', delimiter=' ', newline=' ', header='', footer='')
#     f.write(str(y_train[i].item()))
#     f.write("\n")

    
# f = open("test.txt", "a+")

# f.write(str(len(X_test)))
# f.write(' ')
# f.write(str(len(X_test[0])))
# f.write('\n')
# for i in range(0, len(X_test)):
#     np.savetxt(f, X_test[i], fmt='%1.8f', delimiter=' ', newline=' ', header='', footer='')
#     f.write(str(y_test[i].item()))
#     f.write("\n")



# In[11]:


@jit
def euclidean_distance(vector1, vector2):
    return np.sqrt(np.sum(np.power(vector1-vector2, 2)))


# In[12]:


##Without the parallel version

# @jit
# def get_neighbours(X_train, X_test_instance, k):
#     distances = []
#     neighbors = []
#     #for i in range(0, X_train.shape[0]):
#         dist = euclidean_distance(X_train[i], X_test_instance)
#         distances.append((i, dist))
#     distances.sort(key=operator.itemgetter(1))
#     #for x in range(k):
#         #print distances[x]
#         neighbors.append(distances[x][0])
#     return neighbors


#With the parallel version
@jit(parallel=True)
def get_neighbours(X_train, X_test_instance, k):
    distances = []
    neighbors = []
    for i in prange(0, X_train.shape[0]):
    #for i in range(0, X_train.shape[0]):
        dist = euclidean_distance(X_train[i], X_test_instance)
        distances.append((i, dist))
    distances.sort(key=operator.itemgetter(1))
    #for x in range(k):
    for x in prange(k):
        #print distances[x]
        neighbors.append(distances[x][0])
    return neighbors



# In[13]:


#Without parallel loops
# @jit
# def predictkNNClass(output, y_train):
#     classVotes = {}
#     for i in range(len(output)):
#         if y_train[output[i]] in classVotes:
#             classVotes[y_train[output[i]]] += 1
#         else:
#             classVotes[y_train[output[i]]] = 1
#     sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
#     return sortedVotes[0][0]

#With parallel loop
@jit(parallel=True)
def predictkNNClass(output, y_train):
    classVotes = {}
    for i in prange(len(output)):
        if y_train[output[i]] in classVotes:
            classVotes[y_train[output[i]]] += 1
        else:
            classVotes[y_train[output[i]]] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


# In[14]:


@jit
def kNN_test(X_train, X_test, Y_train, Y_test, k):
    output_classes = []
    for i in range(0, X_test.shape[0]):
        output = get_neighbours(X_train, X_test[i], k)
        predictedClass = predictkNNClass(output, Y_train)
        output_classes.append(predictedClass)
    return output_classes


# In[15]:


@jit
def prediction_accuracy(predicted_labels, original_labels):
    count = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == original_labels[i]:
            count += 1
    #print count, len(predicted_labels)
    return float(count)/len(predicted_labels)


# In[16]:


predicted_classes = {}
final_accuracies = {}

#Not fixing the accuracy
'''
for k in range(1, 21):
    predicted_classes[k] = kNN_test(X_train, X_test, y_train, y_test, k)
    final_accuracies[k] = prediction_accuracy(predicted_classes[k], y_test)
    print(final_accuracies[k])
'''

#Fixing the accuracy
fixed_accuracy = 0.9919354838709677
k = 1
predicted_classes[k] = kNN_test(X_train, X_test, y_train, y_test, k)
final_accuracies[k] = prediction_accuracy(predicted_classes[k], y_test)
#print(final_accuracies[k])

while(final_accuracies[k] != fixed_accuracy and k <= 21):
    k = k+1
    predicted_classes[k] = kNN_test(X_train, X_test, y_train, y_test, k)
    final_accuracies[k] = prediction_accuracy(predicted_classes[k], y_test)
    #print(final_accuracies[k])
    
    
#Making use of cached version
start = time.time()
fixed_accuracy = 0.9919354838709677
k = 1
predicted_classes[k] = kNN_test(X_train, X_test, y_train, y_test, k)
final_accuracies[k] = prediction_accuracy(predicted_classes[k], y_test)
#print(final_accuracies[k])

while(final_accuracies[k] != fixed_accuracy and k <= 21):
    k = k+1
    predicted_classes[k] = kNN_test(X_train, X_test, y_train, y_test, k)
    final_accuracies[k] = prediction_accuracy(predicted_classes[k], y_test)
    #print(final_accuracies[k])
    
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
    
    

