
import sys
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

PATH = "Genes/data.csv"

N_PC = [2,5,10,20,50]
K = 10 #for k fold cross validation
N_NEIGHBOURS = [1,10,30] # for knn classifier


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
confusion_matrix = [[0,0,0,0,0] for x in range(5)]

performance = [0 for x in N_PC]

results = [ [0 for y in range(len(N_PC))] for x in range(len(N_NEIGHBOURS))]
for i, n_components in enumerate(N_PC):

    print("PCA with {0} pcs...".format(n_components))
    pca = PCA(n_components=n_components)
    reduced_data_np = pca.fit(data).transform(data)
    print("...finished pca")

    reduced_data = reduced_data_np.tolist()
    for j,value in enumerate(reduced_data):
        value.append(labels[j])
    random.shuffle(reduced_data) #shuffle data

    for j,n_neigh in enumerate(N_NEIGHBOURS):
        for fold_position in range(K):
            testing  = reduced_data[ int(fold_position * size_training) : int((fold_position +1 ) * size_training)]

            training = reduced_data[: int(fold_position * size_training)] +  reduced_data[int((fold_position+1) * size_training):]
            
            knn = KNeighborsClassifier(n_neighbors=n_neigh)
            knn.fit( [x[:-1] for x in training],  [x[-1] for x in training])


            for value in testing:
                if knn.predict([value[:-1]])[0] == value[-1]:
                    results[j][i] += 1
    
markers = ['o', '^', 's','v', '<','>','1','2','3','4' ]
for i,r in enumerate(results):
    plt.plot(N_PC,[(x*100) / len(data)  for x in r], label = "k = " + str(N_NEIGHBOURS[i]), marker = markers[i] )
    print([x / len(data)  for x in r])

plt.legend()

plt.xlabel("N Principal components")
plt.ylabel("Test accuracy in %")

plt.title("Test accuracy for KNN")

plt.legend()
plt.grid()
plt.show()

