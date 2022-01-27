
import sys
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import numpy as np
from decision_tree import *


PATH = "Genes/data.csv"

#number of principal components
N_PC = [5,6,7,8,9]

#k for k-fold cross validation
K = 10

# number of bins to transform to symbolic data
N_BINS = 4

#Stopping size for branch in decision trees
STOPPING_VALUE = 1


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

distinct_lables = ['PRAD', 'LUAD', 'BRCA', 'KIRC', 'COAD']

for i, n_components in enumerate(N_PC):
    confusion_matrix = [[0,0,0,0,0] for x in range(5)]
    performance = 0

    pca = PCA(n_components=n_components)
    reduced_data_np = pca.fit(data).transform(data)

    reduced_data = reduced_data_np.tolist()
    for j,value in enumerate(reduced_data):
        value.append(labels[j])
    random.shuffle(reduced_data) #shuffle data
    for fold_position in range(K):
        testing  = reduced_data[ int(fold_position * size_training) : int((fold_position +1 ) * size_training)]

        training = reduced_data[: int(fold_position * size_training)] +  reduced_data[int((fold_position+1) * size_training):]
        
        tr = decisionTree(training,bins = N_BINS, stoppingCriteria= "size", stoppingValue= STOPPING_VALUE)
        tr.train()
        for value in testing:
            i1 = distinct_lables.index(value[-1])
            i2 = distinct_lables.index(tr.classify(value[:-1]))
            confusion_matrix[i1][i2] +=1

            if tr.classify(value[:-1]) == value[-1]:
                performance+= 1
    
    performance /= len(reduced_data)

    print("Accuracy using {0} principal components:".format(n_components))
    print(performance*100)
    print("confusion matrix:")
    print(confusion_matrix)





