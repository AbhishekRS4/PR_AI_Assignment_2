import pandas as pd
import scipy.stats
import copy

#last value is class, first value are features
data = [[1,1,1],[1,1,1],[1,0,1],[0,0,1],[1,1,1],[0,1,1],[1,0,0],[1,1,1]]

def calculateEntopy(set):
    data = [vector[len(vector)-1] for vector in set]
    pd_series = pd.Series(data)
    counts = pd_series.value_counts()
    return scipy.stats.entropy(counts)

def splitBranchRec(data, allFeatures):
    minEntropy = float("inf")
    fetureMinEnropy = -1
    valueMinEntropy = -1
    for index, feature in enumerate(allFeatures):
        if len(feature) ==1:
            continue
        for value in feature:
            if len(feature) ==2 and value == feature[1]:
                continue
            dataBranch1 = []
            dataBranch2 = []
            for instance in data:
                if instance[index] == value:
                    dataBranch1.append(instance)
                else:
                    dataBranch2.append(instance)
            entopy = calculateEntopy(dataBranch1) + calculateEntopy(dataBranch2) 
            if entopy < minEntropy:
                minEntropy = entopy
                fetureMinEnropy = index
                valueMinEntropy = value
    if fetureMinEnropy != -1:
        dataBranch1 = []
        dataBranch2 = []
        for instance in data:
            if instance[fetureMinEnropy] == valueMinEntropy:
                dataBranch1.append(instance)
            else:
                dataBranch2.append(instance)
        newAllFeatures = copy.deepcopy(allFeatures)
        newAllFeatures[fetureMinEnropy].remove(valueMinEntropy)
        leftBranch = splitBranchRec(dataBranch1, newAllFeatures)
        rightBranch = splitBranchRec(dataBranch2, newAllFeatures)
        return [fetureMinEnropy,valueMinEntropy, leftBranch,rightBranch]
    else:
        labels = [x[-1] for x in data]
        return # return most common value in labels

allFeatures = [ [] for x in range(len(data[0])-1)]
for vector in data:
    for idx,value in enumerate(vector[0:-1]):
        if value not in allFeatures[idx]:
            allFeatures[idx].append(value)

def classify(vector, tree):
    if tree.isnumeric():
        return tree 
    if vector[tree[0]] == tree[1]:
        #go along left branch
        return classify(vector,tree[2])
    else:
        #go along right branch
        return classify(vector,tree[3])

tree = splitBranchRec(data,allFeatures)
print(tree)




        





        

        


