from decision_tree import *
from random import randrange

class randomForest:
    def __init__(self, data, n_trees = 100, bins = 2, stoppingCriteria = "purity", stoppingValue = 0.95):
        self.trees = []
        for i in range(n_trees):
            bootstrap = self.createBootstrap(data)
            tree = decisionTree(bootstrap,bins,stoppingCriteria,stoppingValue)
            tree.train()
            self.trees.append(tree)

    def createBootstrap(self,data):
        set = []
        for i in range(len(data)):
            set.append(data[randrange(len(data))])
        return set

    def classify(self,vector):
        answers = []
        for tree in self.trees:
            answers.append(tree.classify(vector))
        print(answers)
        return max(set(answers), key = answers.count)

    


data = [[0.9,79,"green"],[0.1,12,"green"],[0.9,10,"red"],[0.2,112.4,"red"]]
forest = randomForest(data,10)

print(forest.classify([0.3,112.4]))




        





        

        


