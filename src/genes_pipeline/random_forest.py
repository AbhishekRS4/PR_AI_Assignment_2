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
        return max(set(answers), key = answers.count)





        





        

        


