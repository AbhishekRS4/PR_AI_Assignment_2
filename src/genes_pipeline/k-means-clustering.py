
import sys
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


PATH = "Genes/data.csv"

#number of principal components
N_PC = [2,3,4,5,6,7]

#k for k-means clustering
K = [2,3,4,5,6,7]



def getData(path):
    data = []
    labels = []
    distinct_labels = []

    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row[1:])

    with open("Genes/labels.csv", newline='') as csvfile:
        reader = csv.reader(csvfile)
        for label in reader:
            try:
                labels.append(label[1])
            except ValueError:
                print(label)
                labels.append(label)

    #not interested in first row with gene description (278820 genes, 802 individuals)
    data = data[1:]
    #transform to floats
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j] = float(data[i][j]) 

    labels = labels[1:]

    return data, labels

print("Loading data...")

data, labels = getData(PATH)

print("...finished loading data")


scores = [[] for x in K]
for n_component in N_PC:
    pca = PCA(n_components=n_component)
    reduced_data = pca.fit(data).transform(data)

    for i,n_clusters in enumerate(K):
        km = KMeans(n_clusters=n_clusters)
        km.fit_predict(reduced_data)
        score = silhouette_score(reduced_data, km.labels_, metric='euclidean')
        scores[i].append(score)

markers = ["o", "^", "<", "s", "x", "1","2,", "3"]
marker_index = 0
for score,n_clusters in zip(scores,K):
    plt.plot(N_PC,score, label = str(n_clusters) + " number of clusters", marker= markers[marker_index])
    marker_index +=1

plt.legend()
plt.xlabel("N Principal Components", fontsize='medium')
plt.ylabel("Silhouette Score", fontsize='medium')
plt.grid()

plt.show()
