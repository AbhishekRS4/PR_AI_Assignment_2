
import sys
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import numpy as np

from decision_tree import *

from random_forest import *

PATH = "Genes/data.csv"

N_PC = [5]
K = 10


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

    distinct_labels = [] #should be ['PRAD', 'LUAD', 'BRCA', 'KIRC', 'COAD']
    for label in labels:
        if label not in distinct_labels:
            distinct_labels.append(label)
    return data, labels

print("Loading data...")

data, labels = getData(PATH)

print("...finished loading data")


training = []
testing = []
size_training = len(data) / K


performance = [0 for x in N_PC]
for i, n_components in enumerate(N_PC):

    print("PCA with {0} pcs...".format(n_components))
    pca = PCA(n_components=n_components)
    reduced_data_np = pca.fit(data).transform(data)
    print("...finished pca")

    reduced_data = reduced_data_np.tolist()
    for j,value in enumerate(reduced_data):
        value.append(labels[j])
    random.shuffle(reduced_data) #shuffle data
    for fold_position in range(K):
        testing  = reduced_data[ int(fold_position * size_training) : int((fold_position +1 ) * size_training)]

        training = reduced_data[: int(fold_position * size_training)] +  reduced_data[int((fold_position+1) * size_training):]
        
        forrest = randomForest(reduced_data,n_trees= 40, bins= 4, stoppingCriteria= "size", stoppingValue= 1)


        for value in testing:
            if forrest.classify(value[:-1]) == value[-1]:
                performance[i] += 1
    
    performance[i] /= len(reduced_data)

print(performance)

#[n_components +1 - x for x in range(n_components)]



