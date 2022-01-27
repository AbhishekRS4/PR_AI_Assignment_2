
from functools import reduce
import sys
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import numpy as np

from decision_tree import *

from random_forest import *
from sklearn.neighbors import KNeighborsClassifier

PATH = "Genes/data.csv"

# nPC, trees, bins, min size
FORESTS_SETTINGS = [[4,10,4,1],[5,10,4,1],[6,10,4,1],[7,10,4,1],[8,10,8,20],[4,10,8,20],[5,10,8,20],[6,10,8,20],[7,10,8,20],[8,10,8,20]]
# nPC, K
KNN_SETTINGS = [[30,1],[30,10],[40,1],[40,10],[50,1],[50,10],[60,1],[60,10],[70,1],[70,10]]
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
    return data, labels

def reduce_data(original_data,n_components, labels):
    pca = PCA(n_components=n_components)
    reduced_data_np = pca.fit(original_data).transform(original_data)

    reduced_data = reduced_data_np.tolist()
    for j,value in enumerate(reduced_data):
        value.append(labels[j]) #append labels
    return reduced_data

print("Loading data...")

data, labels = getData(PATH)

print("...finished loading data")

training = []
testing = []
size_training = len(data) / K


performance = 0


for fold_position in range(K):
    forest_answers = [[] for x in range(len(FORESTS_SETTINGS))]
    knn_answers = [[] for x in range(len(KNN_SETTINGS))]


    for i,forest_attributes in enumerate(FORESTS_SETTINGS):
        reduced_data = reduce_data(data,forest_attributes[0], labels)

        testing  = reduced_data[ int(fold_position * size_training) : int((fold_position +1 ) * size_training)]
        training = reduced_data[: int(fold_position * size_training)] +  reduced_data[int((fold_position+1) * size_training):]


        print("train forest...")
        forest = randomForest(training, n_trees = forest_attributes[1], bins = forest_attributes[2], stoppingCriteria= "size", stoppingValue= forest_attributes[3])
        print("finished traing forest")
        
        for value in testing:
            forest_answers[i].append(forest.classify(value[:-1]))

    for i, knn_attributes in enumerate(KNN_SETTINGS):

        reduced_data = reduce_data(data,knn_attributes[0], labels)
        testing  = reduced_data[ int(fold_position * size_training) : int((fold_position +1 ) * size_training)]
        training = reduced_data[: int(fold_position * size_training)] +  reduced_data[int((fold_position+1) * size_training):]

        print("train knn...")
        knn = KNeighborsClassifier(n_neighbors=knn_attributes[1])
        knn.fit( [x[:-1] for x in training],  [x[-1] for x in training])
        print("...finished traing knn")

        for value in testing:
            knn_answers[i].append(knn.predict([value[:-1]])[0])


    for i,vector in enumerate(testing):
        answers = [f_answer[i] for f_answer in forest_answers] + [knn_answer[i] for knn_answer in knn_answers]
        if max(answers) == vector[-1]:
            performance += 1


performance /= len(data)

print(performance)

#0.9550561797752809

