import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

def clustering(vectors):
    plt.figure(figsize=(10, 7))
    cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    cluster.fit_predict(vectors)
    print(cluster.labels_)
    plt.scatter(vectors[:,0],vectors[:,1], c=cluster.labels_, cmap='rainbow')
    plt.savefig("scatter_clusters.png")

def generate_dengrogram(vectors):
    linked = linkage(vectors, 'ward')
    labelList = range(len(vectors))
    plt.figure(figsize=(10, 7))
    dendrogram(linked,
                orientation='top',
                labels=labelList,
                distance_sort='descending',
                show_leaf_counts=True)
    plt.savefig("dendrogram.png")

def preprocessing(filename):
    vectors = []
    with open(filename, "r") as f:
        for line in f:
            temp = []
            for d in line.strip().split():
                temp.append(float(d))
            vectors.append(temp)
    return vectors

def plot_points(vectors):
    labels = range(len(vectors))
    plt.figure(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.1)
    plt.scatter(vectors[:,0],vectors[:,1], label='True Position')

    for label, x, y in zip(labels, vectors[:, 0], vectors[:, 1]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-3, 3),
            textcoords='offset points', ha='right', va='bottom')
    plt.show()

if __name__ == "__main__":

    filename = "data.txt"
    vectors = preprocessing(filename)
    vectors = np.asarray(vectors)
    # plot_points(vectors)
    generate_dengrogram(vectors)
    clustering(vectors)

