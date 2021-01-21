import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from decision_tree import DecisionTree

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def evaluate(y,yhat):
#     y =y.astype('int32') 
#     yhat = yhat.astype('int32')
 
    matrix_test = confusion_matrix(y, yhat)
    
    TP = matrix_test[0][0]
    FN = matrix_test[0][1]
    FP = matrix_test[1][0]
    TN = matrix_test[1][1]

    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    accuracy = (TP+TN)/(TP+TN+FN+FP)
    print("confusion matrix",matrix_test)
    return sensitivity,specificity,accuracy

def Decision_Tree_Model(X,Y,test_split_ratio):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    senst,spec,acc = evaluate(y_test, y_pred)

    return senst,spec,acc