
import sys
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = []
labels = []

with open("Genes/data.csv", newline='') as csvfile:
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

pca = PCA(n_components=500)
reduced_data = pca.fit(data).transform(data)

expl_variance = pca.explained_variance_ratio_
print(expl_variance[10])
print(sum(expl_variance[:10]))
print(expl_variance[100])
print(sum(expl_variance[:100]))
print(expl_variance[499])
print(sum(expl_variance))


if False:
    #create plot with classes over two PC
    pca = PCA(n_components=2)
    reduced_data = pca.fit(data).transform(data)

    print(
        "explained variance ratio (first two components): %s"
        % str(pca.explained_variance_ratio_)
    )

    print(len(reduced_data))
    print(len(reduced_data[0]))


    for label in distinct_labels:
        x = [value[0] for i,value in enumerate(reduced_data) if labels[i] == label]
        y = [value[1] for i,value in enumerate(reduced_data) if labels[i] == label]
        print("{0} values for class ".format(len(x)) + label)
        plt.scatter(x,y,label = label)

    plt.xlabel("First PC")
    plt.ylabel("Second PC")

    plt.title("Classes with two PC")

    plt.legend()
    plt.show()
