import pandas as pd
import scipy.stats
import copy
import sys
import random

class decisionTree:

    def __init__(self, data, bins = 2, stoppingCriteria = "purity", stoppingValue = 0.95):
        if stoppingCriteria == "purity":
            if stoppingValue > 1 or stoppingValue < 0:
                print("purity of leaves should be between 0 or 1") 
                sys.exit(1)
            self.stoppingCriteria = "purity"
            self.stoppingValue = stoppingValue
        elif stoppingCriteria == "size":
            self.stoppingCriteria = "size"
            self.stoppingValue = stoppingValue
        else:
            print("stoppingCriteria should be 'size' or 'purity'") 
            sys.exit(1)

        self.n_features = len(data[0]) -1

        if isinstance(bins,int):
            x = bins
            bins = [x for i in range(self.n_features)]
        elif isinstance(bins,list):
            if len(bins) > self.n_features:
                print("List of bins is longer than number of features") 
                sys.exit(1)
            while len(bins) < self.n_features :
                bins.append(2)
        else:
            print("Bin variable has wrong type") 
            sys.exit(1)

        self.classIntMaping = {}
        self.tree = None
        self.values_per_bin = [len(data) / bins[i] for i in range(self.n_features)]
        self.bins = bins
        self.originalData = data
        self.thresholds = [[]for x in range(self.n_features)]
        self.data = self.dataToSymbolic(self.originalData)


        self.allFeatures = [ [] for x in range(self.n_features)]
        for vector in self.data:
            for idx,value in enumerate(vector[:-1]):
                if value not in self.allFeatures[idx]:
                    self.allFeatures[idx].append(value)
    
    def train(self):
        self.tree = self.splitBranchRec(self.data,self.allFeatures)

    def  dataToSymbolic(self,data):
        symbolic_data = [[] for i in range(len(data))]
        
        for idx_feature in range(self.n_features):
            values = []
            for vector in data:
                values.append(vector[idx_feature])
            values = sorted(values)
            
            self.thresholds[idx_feature] = [values[ (1+ x) * ( (int)(self.values_per_bin[idx_feature]) -1 )  ] for x in range(self.bins[idx_feature])]
                
                

            
            for i,vector in enumerate(data):
                bin =0

                while vector[idx_feature] > self.thresholds[idx_feature][bin]:
                    bin +=1
                    if bin == len(self.thresholds[idx_feature]):
                        break

                symbolic_data[i].append(bin)
            
        n_classes = 0
        for i,vector in enumerate(data):
            if vector[-1] not in self.classIntMaping:
                self.classIntMaping[vector[-1]] = n_classes
                n_classes +=1
            symbolic_data[i].append(self.classIntMaping.get(vector[-1]))
        return symbolic_data

    def calculateEntropy(self,set):
        data = [vector[len(vector)-1] for vector in set]
        pd_series = pd.Series(data,dtype='float64')
        counts = pd_series.value_counts()
        return scipy.stats.entropy(counts)

    def splitBranchRec(self,currentData,features):

        labels = [x[-1] for x in currentData]
        purity = labels.count( max(set(labels), key=labels.count) ) / len(labels)

        #stopping criteria
        if purity == 1 or \
                (self.stoppingCriteria == "size" and self.stoppingValue >= len(currentData)) or\
                (self.stoppingCriteria == "purity" and self.stoppingValue < purity):
            return max(set(labels), key=labels.count)

        minEntropy = float("inf")
        featureMinEntropy = -1
        valueMinEntropy = -1


        for index, feature in enumerate(features): 
            if len(feature) ==1:
                continue

            #if random.randint(0,100) != 42:
            #    continue

            for value in feature:
                if len(feature) ==2 and value == feature[1]:
                    #already investigated this option of splitting
                    continue

                dataBranch1 = []
                dataBranch2 = []
                for instance in currentData:
                    if instance[index] == value:
                        dataBranch1.append(instance)
                    else:
                        dataBranch2.append(instance)

                if len(dataBranch1) == 0 or len(dataBranch2) == 0:
                    continue
                entropy = len(dataBranch1) * self.calculateEntropy(dataBranch1) + len(dataBranch1) * self.calculateEntropy(dataBranch2) 
                if entropy < minEntropy:
                    minEntropy = entropy
                    featureMinEntropy = index
                    valueMinEntropy = value

        if featureMinEntropy != -1:
            dataBranch1 = []
            dataBranch2 = []
            for instance in currentData:
                if instance[featureMinEntropy] == valueMinEntropy:
                    dataBranch1.append(instance)
                else:
                    dataBranch2.append(instance)
            newFeatures = copy.deepcopy(self.allFeatures)
            newFeatures[featureMinEntropy].remove(valueMinEntropy)
            leftBranch = self.splitBranchRec(dataBranch1, newFeatures)
            rightBranch = self.splitBranchRec(dataBranch2, newFeatures)
            return [featureMinEntropy,valueMinEntropy, leftBranch,rightBranch]
        else:
            #there is no feature that can split the data at this point
            return max(set(labels), key=labels.count)

    def classifyRec(self, currentTree,vector):

        if isinstance(currentTree,int):
            return list(self.classIntMaping.keys())[ list(self.classIntMaping.values()).index(currentTree)]
        if vector[currentTree[0]] == currentTree[1]:
            #go along left branch
            return self.classifyRec(currentTree[2],vector)
        else:
            #go along right branch
            return self.classifyRec(currentTree[3],vector)

    def classify(self,vector):
        if self.tree == None:
            print("Tree is not trained jet (use 'train()'") 
            sys.exit(1)

        #to symbolic data:
        symbolic_vector = []
        for i,value in enumerate(vector):
            bin =0
            while value > self.thresholds[i][bin]:
                bin +=1
                if bin >= len(self.thresholds[i]):
                    break



            symbolic_vector.append(bin)

        return self.classifyRec(self.tree,symbolic_vector )



#data = [[0.5,42,"green"],[0.1,63,"green"],[0.9,80,"blue"],[0.3,112.4,"red"]]
#tree = decisionTree(data,[2,2],stoppingCriteria='size',stoppingValue= 2)
#tree.train()
#print(tree.tree)
#print(tree.classify([0.32,130]))




        





        

        


