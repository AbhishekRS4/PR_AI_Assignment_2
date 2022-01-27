import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from matplotlib import pyplot as p


# simple function to replace labels by -1 in x_train_unlab
def minus(x):
    return -1

numpy.seterr(divide='ignore', invalid='ignore')

# read data from csv file as a panda data frame
data = pd.read_csv("creditcard.csv")

# removing the unnecessary columns from the data set
data = data.drop('Amount', axis = 1)
data = data.drop('Time', axis = 1)

# creating empty lists for the accuracy and F1 scores that can be filled over all 100 iterations to calculate the means
# and plot the accuracy over multiple iterations
accuracy_lr =  []
f1_score_lr = []
accuracy_lp =[]
f1_score_lp = []
accuracy_lrlp = []
f1_score_lrlp = []

# repeat 100 times
for x in range(100):

    # shuffle all the data
    shuffled_data = data.sample(frac=1)

    # put all the data with class = 1 in a separate data set.
    fraudulent_data = shuffled_data.loc[shuffled_data['Class'] == 1]

    # randomly sample which is the same size as the fraudulent data set
    normal_data = shuffled_data.loc[shuffled_data['Class'] == 0].sample(n=len(fraudulent_data))

    # concatenate both data frames to form a single data set
    data = pd.concat([fraudulent_data, normal_data])
    data = data.sample(frac=1,random_state=1)

    # create training data set as 80% of total data set
    x_train = data.sample(frac = 0.8)

    # remaining 20% is test data
    x_test = data.drop(x_train.index)

    # divide training data into 30% labelled data and 70% unlabelled data
    x_train_lab = x_train.sample(frac=0.3)
    x_train_unlab = x_train.drop(x_train_lab.index)

    # calculate the ratio of fraudulent transactions (data labelled with Class = 1) in the training and testing data sets to
    # see if the distribution between the different classes for both data sets is approximately equal
    fraudulent = len(x_train[x_train.Class == 1])
    fraud_percentage_train = round(fraudulent/len(x_train)*100, 2)
    fraudulent = len(x_test[x_test.Class == 1])
    fraud_percentage_test = round(fraudulent/len(x_test)*100, 2)

    # # print the amounts of entries in each subset and the percentage of fraudulent transactions
    # print("x_train", len(x_train), fraud_percentage_train)
    # print("x_test", len(x_test), fraud_percentage_test)
    # print("x_train_lab", len(x_train_lab))
    # print("x_train_unlab", len(x_train_unlab))

    # make the data sets X and y for logistic regression
    X_train_lr = x_train_lab.drop('Class', axis = 1).values
    y_train_lr = x_train_lab['Class'].values
    X_test = x_test.drop('Class', axis = 1).values
    y_test = x_test['Class'].values

    # train and test the logistic regression model
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train_lr, y_train_lr)
    lr_test = lr.predict(X_test)

    accuracy_lr.append(accuracy_score(y_test, lr_test))
    f1_score_lr.append(f1_score(y_test, lr_test))

    # make the data sets X and y for label propagation
    # append x_train_lab to x_train_unlab to form a dataset x_train with the correct indices
    x_frames = [x_train_lab, x_train_unlab]
    X_train_lp = pd.concat(x_frames)
    X_train_lp = X_train_lp.drop('Class', axis = 1).values

    # create dataframe for the labels in x_train_lab and a separate dataframe that replaces all labels in x_train_unlab
    # with -1. Append the two dataframes
    y_train_lab = x_train_lab['Class']
    y_train_unlab = x_train_unlab["Class"].apply(minus)
    y_frames = [y_train_lab, y_train_unlab]
    y_train_lp = pd.concat(y_frames).values

    # train and test the label propagation model
    lp = LabelPropagation()
    lp.fit(X_train_lp, y_train_lp)
    lp_test = lp.predict(X_test)

    accuracy_lp.append(accuracy_score(y_test, lp_test))
    f1_score_lp.append(f1_score(y_test, lp_test))

    # get the transduction of the Label Propagation model
    y_train_lrlp = lp.transduction_

    # use the transduction as the target values for the Logistic Regression model
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train_lp, y_train_lrlp)
    lrlp_test = lr.predict(X_test)

    accuracy_lrlp.append(accuracy_score(y_test, lrlp_test))
    f1_score_lrlp.append(f1_score(y_test, lrlp_test))

# print the average accuracies and f1 scores of all the models
print('Average accuracy of the Logistic Regression model: ', sum(accuracy_lr)/len(accuracy_lr))
print('Average F1 score of the Logistic Regression model: ', sum(f1_score_lr)/len(f1_score_lr))

print(" ")

print('Average accuracy of the Label Propagation model: ', sum(accuracy_lp)/len(accuracy_lp))
print('Average F1 score of the Label Propagation model: ', sum(f1_score_lp)/len(f1_score_lp))

print(" ")

print('Average accuracy of the Logistic Regression model with Labels: ', sum(accuracy_lrlp)/len(accuracy_lrlp))
print('Average F1 score of the Logistic Regression model with labels: ', sum(f1_score_lrlp)/len(f1_score_lrlp))

# plot the accuracy scores of all cases over 100 iterations
p.plot(range(100), accuracy_lr)
p.title('accuracy score over 100 runs of Logistic Regression model')
p.xlabel('iteration')
p.ylabel('accuracy score')
p.show()

p.plot(range(100), accuracy_lp)
p.title('accuracy score over 100 runs of Label Propagation model')
p.xlabel('iteration')
p.ylabel('accuracy score')
p.show()

p.plot(range(100), accuracy_lrlp)
p.title('accuracy score over 100 runs of Logistic Regression model with learned labels')
p.xlabel('iteration')
p.ylabel('accuracy score')
p.show()
